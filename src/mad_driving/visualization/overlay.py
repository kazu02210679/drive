"""Offline episode GIF rendering from strict records and persisted lossless frames."""

from __future__ import annotations

import io
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Final

from PIL import Image, ImageDraw, ImageFont

from mad_driving.atomic import rename_no_replace
from mad_driving.control.actions import DrivingAction
from mad_driving.evaluation.models import EvaluationStepRecord
from mad_driving.evaluation.serialization import read_jsonl_strict
from mad_driving.visualization import (
    SMOKE_RESULT_LABEL,
    _find_and_verify_bundle,
    _require_regular_directory,
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
    relative = PurePosixPath(frame_path)
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


def write_episode_gif(step_jsonl: Path, frames_dir: Path, output_gif: Path) -> None:
    """Render a GIF without importing or rerunning a simulator or policy."""

    output = Path(output_gif)
    if output.exists():
        raise FileExistsError(output)
    trace = Path(step_jsonl)
    frame_root = Path(frames_dir)
    bundle = _find_and_verify_bundle(trace)
    _require_regular_directory(frame_root, "frames_dir")
    absolute_frames = Path(os.path.abspath(frame_root))
    try:
        absolute_frames.relative_to(bundle.root)
    except ValueError as error:
        raise ValueError("frames_dir is outside the verified bundle") from error

    records = read_jsonl_strict(trace, EvaluationStepRecord)
    bundle.read_bytes(trace)
    rendered: list[Image.Image] = []
    seen_paths: set[str] = set()
    for record in records:
        path = _frame_path(bundle.root, absolute_frames, record.frame_path)
        relative = bundle.relative_artifact(path)
        if relative in seen_paths:
            raise ValueError(f"step records reuse a persisted frame: {relative}")
        seen_paths.add(relative)
        frame = _decode_lossless_frame(bundle.read_bytes(path), relative)
        rendered.append(_render_overlay(frame, _overlay_lines(record)))
    _write_gif(tuple(rendered), output)


__all__ = ["write_episode_gif"]
