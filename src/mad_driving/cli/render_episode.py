"""Render one persisted episode offline from a verified evaluation bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from mad_driving.cli._common import concise_operational_error
from mad_driving.config.models import MethodId
from mad_driving.evaluation.models import (
    EvaluationEpisodeKey,
    EvaluationStepRecord,
    EvaluationTrack,
    ScenarioCellId,
)
from mad_driving.evaluation.serialization import parse_jsonl_bytes_strict
from mad_driving.visualization import (
    _reject_output_in_source_bundle,
    _VerifiedBundle,
    _verify_bundle,
)

_EPISODE_KEY_PATTERN = re.compile(r"[a-z0-9_]{1,240}\Z")
_TRACE_PATTERN = re.compile(r"episode_([0-9]+)_trace\.jsonl\Z")
_FRAME_PATTERN = re.compile(r"([0-9]{6})\.png\Z")


@dataclass(frozen=True)
class VerifiedArtifactPayload:
    """Manifest identity and handle-verified immutable bytes for one artifact."""

    relative_path: str
    size_bytes: int
    sha256: str
    payload: bytes


@dataclass(frozen=True)
class VerifiedEpisodeRenderInputs:
    """All authenticated inputs needed to render one persisted episode offline."""

    episode_key: EvaluationEpisodeKey
    records: tuple[EvaluationStepRecord, ...]
    trace: VerifiedArtifactPayload
    frames: tuple[VerifiedArtifactPayload, ...]


def run_render_bundle(
    *,
    evaluation: Path,
    render_inputs: VerifiedEpisodeRenderInputs,
    destination: Path,
) -> Path:
    """Render one authenticated trace/frame set into a fresh manifest bundle."""

    from mad_driving.evaluation.bundle import _cleanup_workspace
    from mad_driving.evaluation.workspace import EvaluationWorkspace
    from mad_driving.visualization import write_episode_gif

    workspace: EvaluationWorkspace | None = None
    try:
        workspace = EvaluationWorkspace.stage(destination)
        trace = evaluation / PurePosixPath(render_inputs.trace.relative_path)
        first_frame = evaluation / PurePosixPath(render_inputs.frames[0].relative_path)
        frames = first_frame.parent
        output = workspace.path / "render.gif"
        write_episode_gif(trace, frames, output)
        workspace.write_manifest()
        return workspace.publish()
    except BaseException:
        _cleanup_workspace(workspace)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, help="Verified evaluation bundle")
    parser.add_argument("--episode-key", required=True, help="Exact persisted episode key")
    parser.add_argument(
        "--output",
        help="Fresh output directory; defaults to an episode-specific source sibling",
    )
    return parser


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path)))


def _destination(evaluation: Path, episode_key: str, supplied: str | None) -> Path:
    if supplied is not None:
        return _absolute(supplied)
    return evaluation.with_name(f"{evaluation.name}-render-{episode_key}")


def _canonical_non_negative_int(value: str, label: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"persisted episode {label} is not a canonical integer")
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError(f"persisted episode {label} is not canonical")
    return parsed


def _path_episode_key(parts: tuple[str, ...], seed_text: str) -> EvaluationEpisodeKey:
    method_id, track, policy_text, case_id = parts[1:5]
    seed = _canonical_non_negative_int(seed_text, "seed")
    if policy_text == "rule":
        policy_seed = None
    else:
        policy_seed = _canonical_non_negative_int(policy_text, "policy seed")
    return EvaluationEpisodeKey(
        method_id=cast(MethodId, method_id),
        track=cast(EvaluationTrack, track),
        role="test",
        policy_seed=policy_seed,
        case_id=cast(ScenarioCellId, case_id),
        episode_rng_seed=seed,
    )


def _episode_key_text(key: EvaluationEpisodeKey) -> str:
    policy_seed = "rule" if key.policy_seed is None else str(key.policy_seed)
    return "_".join(
        (
            key.method_id,
            key.track,
            policy_seed,
            key.case_id,
            str(key.episode_rng_seed),
        )
    )


def _artifact_payload(bundle: _VerifiedBundle, relative: str) -> VerifiedArtifactPayload:
    identity = bundle.artifacts[relative]
    payload = bundle.read_bytes(bundle.root / PurePosixPath(relative))
    return VerifiedArtifactPayload(
        relative_path=relative,
        size_bytes=identity.size_bytes,
        sha256=identity.sha256,
        payload=payload,
    )


def _verified_render_inputs(
    bundle: _VerifiedBundle, episode_key: str
) -> VerifiedEpisodeRenderInputs:
    if _EPISODE_KEY_PATTERN.fullmatch(episode_key) is None:
        raise ValueError("episode key contains invalid characters or length")
    matches: list[tuple[str, str, EvaluationEpisodeKey]] = []
    artifact_names = tuple(bundle.artifacts)
    for relative in artifact_names:
        parts = PurePosixPath(relative).parts
        if len(parts) != 6 or parts[0] != "episodes":
            continue
        trace_match = _TRACE_PATTERN.fullmatch(parts[5])
        if trace_match is None:
            continue
        path_key = _path_episode_key(parts, trace_match.group(1))
        if _episode_key_text(path_key) != episode_key:
            continue
        frames_relative = "/".join((*parts[:5], f"episode_{path_key.episode_rng_seed}_frames"))
        matches.append((relative, frames_relative, path_key))
    if len(matches) != 1:
        raise ValueError(
            f"episode key must identify exactly one persisted trace/frame set; found {len(matches)}"
        )
    trace_relative, frames_relative, path_key = matches[0]
    trace = _artifact_payload(bundle, trace_relative)
    records = parse_jsonl_bytes_strict(trace.payload, EvaluationStepRecord)
    if records[0].episode_key != path_key:
        raise ValueError("persisted trace episode key does not match its canonical path")

    frame_prefix = f"{frames_relative}/"
    manifested_frames = tuple(
        relative for relative in artifact_names if relative.startswith(frame_prefix)
    )
    expected_frames = tuple(f"{frames_relative}/{record.step_index:06d}.png" for record in records)
    if manifested_frames != expected_frames:
        raise ValueError(
            "persisted frame inventory must be exact, nonempty, and contiguous by step"
        )
    if any(
        _FRAME_PATTERN.fullmatch(PurePosixPath(relative).name) is None
        for relative in manifested_frames
    ):
        raise ValueError("persisted frame inventory contains an invalid frame name")
    for record, expected in zip(records, expected_frames, strict=True):
        if record.frame_path != expected:
            raise ValueError("persisted trace frame path does not match its manifested step")
    frames = tuple(_artifact_payload(bundle, relative) for relative in manifested_frames)
    return VerifiedEpisodeRenderInputs(
        episode_key=path_key,
        records=records,
        trace=trace,
        frames=frames,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point that selects only manifested persisted episode data."""

    args = _parser().parse_args(argv)
    try:
        evaluation = _absolute(args.evaluation)
        if _EPISODE_KEY_PATTERN.fullmatch(args.episode_key) is None:
            raise ValueError("episode key contains invalid characters or length")
        destination = _destination(evaluation, args.episode_key, args.output)
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        bundle = _verify_bundle(evaluation)
        _reject_output_in_source_bundle(bundle.root, destination)
        render_inputs = _verified_render_inputs(bundle, args.episode_key)
        published = run_render_bundle(
            evaluation=bundle.root,
            render_inputs=render_inputs,
            destination=destination,
        )
        output = {"output": str(published)}
    except Exception as exc:
        print(f"render failed: {concise_operational_error(exc)}", file=sys.stderr)
        return 2
    print(json.dumps(output, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
