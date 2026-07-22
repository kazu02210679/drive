"""Render one persisted episode offline from a verified evaluation bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from mad_driving.visualization import (
    _reject_output_in_source_bundle,
    _VerifiedBundle,
    _verify_bundle,
)

_EPISODE_KEY_PATTERN = re.compile(r"[a-z0-9_]{1,240}\Z")
_TRACE_PATTERN = re.compile(r"episode_([0-9]+)_trace\.jsonl\Z")


def run_render_bundle(
    *,
    evaluation: Path,
    episode_key: str,
    step_jsonl: Path,
    frames_dir: Path,
    destination: Path,
) -> Path:
    """Injected Task 10 offline render orchestration seam."""

    del evaluation, episode_key, step_jsonl, frames_dir, destination
    raise RuntimeError("render bundle orchestration is not installed")


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


def _episode_artifacts(bundle: _VerifiedBundle, episode_key: str) -> tuple[Path, Path]:
    if _EPISODE_KEY_PATTERN.fullmatch(episode_key) is None:
        raise ValueError("episode key contains invalid characters or length")
    matches: list[tuple[str, str]] = []
    artifact_names = frozenset(bundle.artifacts)
    for relative in sorted(artifact_names):
        parts = PurePosixPath(relative).parts
        if len(parts) != 6 or parts[0] != "episodes":
            continue
        trace_match = _TRACE_PATTERN.fullmatch(parts[5])
        if trace_match is None:
            continue
        method_id, track, policy_seed, case_id = parts[1:5]
        seed = trace_match.group(1)
        candidate_key = "_".join((method_id, track, policy_seed, case_id, seed))
        if candidate_key != episode_key:
            continue
        frames_relative = "/".join((*parts[:5], f"episode_{seed}_frames"))
        frame_prefix = f"{frames_relative}/"
        if not any(name.startswith(frame_prefix) for name in artifact_names):
            continue
        matches.append((relative, frames_relative))
    if len(matches) != 1:
        raise ValueError(
            "episode key must identify exactly one persisted trace/frame set; "
            f"found {len(matches)}"
        )
    trace_relative, frames_relative = matches[0]
    return (
        bundle.root / PurePosixPath(trace_relative),
        bundle.root / PurePosixPath(frames_relative),
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
        step_jsonl, frames_dir = _episode_artifacts(bundle, args.episode_key)
        published = run_render_bundle(
            evaluation=bundle.root,
            episode_key=args.episode_key,
            step_jsonl=step_jsonl,
            frames_dir=frames_dir,
            destination=destination,
        )
        output = {"output": str(published)}
    except Exception as exc:
        print(f"render failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
