"""Offline episode GIF rendering from strict records and persisted lossless frames."""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, cast

from PIL import Image, ImageDraw, ImageFont

from mad_driving.atomic import rename_no_replace
from mad_driving.control.actions import DrivingAction
from mad_driving.evaluation.models import EvaluationStepRecord
from mad_driving.visualization import (
    SMOKE_RESULT_LABEL,
    _find_and_verify_bundle,
    _reject_output_in_source_bundle,
    _require_regular_directory,
    _unique_json_object,
)

_FRAME_DURATION_MS: Final = 100
_PANEL_BACKGROUND: Final = (8, 12, 18)
_TEXT_COLOR: Final = (245, 247, 250)
_ACCENT_COLOR: Final = (255, 196, 0)


def _action_label(value: int) -> str:
    try:
        name = DrivingAction(value).name
    except ValueError as error:
        raise ValueError(f"persisted action is outside the fixed action order: {value}") from error
    return f"{name} ({value})"


def _overlay_lines(record: EvaluationStepRecord) -> tuple[str, ...]:
    """Build stable visible text solely from one persisted strict step record."""

    key = record.episode_key
    ttc = "N/A" if record.minimum_actual_ttc_s is None else f"{record.minimum_actual_ttc_s:.2f} s"
    reasons = ", ".join(record.shield_reasons) if record.shield_reasons else "none"
    shield = "INTERVENED" if record.shield_intervened else "no intervention"
    lines = [
        f"Method: {key.method_id}",
        f"Track: {key.track}",
        f"Scenario: {record.case_id} ({record.scenario_id})",
        f"Seed: {record.episode_rng_seed}",
        f"Step: {record.step_index}",
        f"Requested: {_action_label(record.requested_action)}",
        f"Required: {_action_label(record.required_action)}",
        f"Executed: {_action_label(record.executed_action)}",
        f"Speed: {record.ego_speed_mps:.2f} m/s",
        f"Oracle TTC: {ttc}",
        f"Speed limit: {record.ego_speed_limit_mps:.2f} m/s",
        f"Target speed: {record.target_speed_mps:.2f} m/s",
        f"Shield: {shield} [{reasons}] (mode={record.shield_mode})",
    ]
    claims_by_agent: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for claim in record.claims:
        claims_by_agent[claim.agent_id].append((claim.severity, claim.recommended_max_speed_mps))
    for agent_id in record.expected_agent_ids:
        claims = claims_by_agent.get(agent_id, ())
        if claims:
            lines.append(
                f"Agent {agent_id}: severity={max(item[0] for item in claims):.2f}, "
                f"recommended speed={min(item[1] for item in claims):.2f} m/s"
            )
        else:
            lines.append(f"Agent {agent_id}: severity=N/A, recommended speed=N/A")
    lines.append(f"Critic conflict: {record.review.conflict_score:.2f}")
    if not record.is_formal:
        lines.append(SMOKE_RESULT_LABEL)
    return tuple(lines)


def _frame_path(bundle_root: Path, frames_dir: Path, frame_path: str | None) -> Path:
    if frame_path is None:
        raise ValueError("every rendered step record must declare a persisted frame_path")
    if "\\" in frame_path or "//" in frame_path:
        raise ValueError("frame_path must use canonical POSIX relative spelling")
    relative = PurePosixPath(frame_path)
    windows = PureWindowsPath(frame_path)
    if (
        relative.as_posix() != frame_path
        or relative.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "." in relative.parts
        or ".." in relative.parts
    ):
        raise ValueError("frame_path must use canonical POSIX relative spelling")
    candidate = Path(os.path.abspath(bundle_root / relative))
    frame_root = Path(os.path.abspath(frames_dir))
    try:
        candidate.relative_to(frame_root)
    except ValueError as error:
        raise ValueError(f"record frame_path is outside frames_dir: {frame_path}") from error
    return candidate


def _decode_lossless_frame(payload: bytes, relative: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format != "PNG":
                raise ValueError(f"persisted frame must be a lossless PNG: {relative}")
            source.load()
            frame = source.convert("RGB")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"persisted frame is unreadable: {relative}") from error
    if frame.width < 1 or frame.height < 1:
        raise ValueError(f"persisted frame has invalid dimensions: {relative}")
    return frame


def _render_overlay(frame: Image.Image, lines: tuple[str, ...]) -> Image.Image:
    font = ImageFont.load_default()
    probe = ImageDraw.Draw(frame)
    line_height = max(11, math.ceil(probe.textbbox((0, 0), "Ag", font=font)[3] + 2))
    panel_height = 10 + line_height * len(lines)
    rendered = Image.new("RGB", (frame.width, frame.height + panel_height), _PANEL_BACKGROUND)
    rendered.paste(frame, (0, 0))
    draw = ImageDraw.Draw(rendered)
    y = frame.height + 5
    for line in lines:
        color = _ACCENT_COLOR if line == SMOKE_RESULT_LABEL else _TEXT_COLOR
        draw.text((7, y), line, font=font, fill=color)
        y += line_height
    return rendered


def _write_gif(frames: tuple[Image.Image, ...], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".gif", dir=destination.parent, delete=False
        ).name
    )
    try:
        first, *remaining = frames
        first.save(
            temporary,
            format="GIF",
            save_all=True,
            append_images=remaining,
            duration=_FRAME_DURATION_MS,
            loop=0,
            optimize=False,
            disposal=2,
            comment=b"mad-driving verified persisted frames",
        )
        rename_no_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _read_step_records(payload: bytes) -> tuple[EvaluationStepRecord, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("step JSONL file is not UTF-8") from error
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("step JSONL file must be non-empty and end with a trailing newline")
    lines = text.splitlines()
    if any(not line for line in lines):
        raise ValueError("step JSONL file contains a blank record")
    records: list[EvaluationStepRecord] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"step JSONL record is malformed at line {line_number}") from error
        if not isinstance(raw, Mapping):
            raise ValueError(f"step JSONL record must be an object at line {line_number}")
        try:
            records.append(EvaluationStepRecord.from_dict(cast(Mapping[str, object], raw)))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"step JSONL record is invalid at line {line_number}: {error}"
            ) from error
    result = tuple(records)
    first = result[0]
    if any(record.episode_key != first.episode_key for record in result[1:]):
        raise ValueError("step JSONL file contains more than one episode key")
    if any(record.episode_index != first.episode_index for record in result[1:]):
        raise ValueError("step JSONL file contains more than one episode_index")
    if any(record.is_formal is not first.is_formal for record in result[1:]):
        raise ValueError("step JSONL file contains more than one is_formal value")
    if any(record.shield_mode != first.shield_mode for record in result[1:]):
        raise ValueError("step JSONL file contains more than one shield_mode")
    if any(record.step_index != expected for expected, record in enumerate(result)):
        raise ValueError("step indices must be contiguous and zero-based")
    return result


def write_episode_gif(step_jsonl: Path, frames_dir: Path, output_gif: Path) -> None:
    """Render a GIF without importing or rerunning a simulator or policy."""

    output = Path(output_gif)
    trace = Path(step_jsonl)
    frame_root = Path(frames_dir)
    bundle = _find_and_verify_bundle(trace)
    _reject_output_in_source_bundle(bundle.root, output)
    if output.exists():
        raise FileExistsError(output)
    _require_regular_directory(frame_root, "frames_dir")
    absolute_frames = Path(os.path.abspath(frame_root))
    try:
        absolute_frames.relative_to(bundle.root)
    except ValueError as error:
        raise ValueError("frames_dir is outside the verified bundle") from error

    records = _read_step_records(bundle.read_bytes(trace))
    frames_relative = absolute_frames.relative_to(bundle.root).as_posix()
    expected_frames = tuple(f"{frames_relative}/{record.step_index:06d}.png" for record in records)
    declared_frames = tuple(
        relative for relative in bundle.artifacts if relative.startswith(f"{frames_relative}/")
    )
    if declared_frames != expected_frames:
        raise ValueError(
            "declared frame inventory must exactly match contiguous step-index frame identities"
        )
    rendered: list[Image.Image] = []
    seen_paths: set[str] = set()
    for record, expected_relative in zip(records, expected_frames, strict=True):
        if record.frame_path != expected_relative:
            raise ValueError(
                "record frame_path must stay inside frames_dir and match its "
                "deterministic step-index frame identity"
            )
        path = _frame_path(bundle.root, absolute_frames, record.frame_path)
        relative = bundle.relative_artifact(path)
        if relative in seen_paths:
            raise ValueError(f"step records reuse a persisted frame: {relative}")
        seen_paths.add(relative)
        frame = _decode_lossless_frame(bundle.read_bytes(path), relative)
        rendered.append(_render_overlay(frame, _overlay_lines(record)))
    _write_gif(tuple(rendered), output)


__all__ = ["write_episode_gif"]
