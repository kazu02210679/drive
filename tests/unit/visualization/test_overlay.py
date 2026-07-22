from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from mad_driving.evaluation.workspace import EvaluationWorkspace
from mad_driving.visualization.overlay import _overlay_lines, write_episode_gif

from .conftest import make_step


def _rewrite_trace_and_manifest(
    bundle: Path, trace: Path, payloads: list[dict[str, object]]
) -> None:
    (bundle / "evaluation_manifest.json").unlink()
    trace.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in payloads),
        encoding="utf-8",
    )
    EvaluationWorkspace(destination=bundle, path=bundle).write_manifest()


def test_overlay_contains_every_required_persisted_agent_and_critic_field() -> None:
    lines = _overlay_lines(make_step(step_index=0, frame_path="frames/000000.png"))
    text = "\n".join(lines)

    for expected in (
        "Method: proposed",
        "Track: system",
        "Scenario: level1_lead_brake (lead_brake)",
        "Seed: 20001",
        "Step: 0",
        "Requested: SLOW (1)",
        "Required: PREPARE_STOP (2)",
        "Executed: PREPARE_STOP (2)",
        "Speed: 10.00 m/s",
        "Oracle TTC: 2.40 s",
        "Speed limit: 13.00 m/s",
        "Target speed: 8.00 m/s",
        "Shield: INTERVENED [minimum_ttc] (mode=enforce)",
        "Agent nominal: severity=0.20, recommended speed=12.00 m/s",
        "Agent hazard: severity=0.80, recommended speed=4.00 m/s",
        "Agent rule: severity=0.50, recommended speed=6.00 m/s",
        "Critic conflict: 0.65",
    ):
        assert expected in text


def test_overlay_text_has_one_exact_deterministic_order() -> None:
    assert _overlay_lines(make_step(step_index=0, frame_path="frames/000000.png")) == (
        "Method: proposed",
        "Track: system",
        "Scenario: level1_lead_brake (lead_brake)",
        "Seed: 20001",
        "Step: 0",
        "Requested: SLOW (1)",
        "Required: PREPARE_STOP (2)",
        "Executed: PREPARE_STOP (2)",
        "Speed: 10.00 m/s",
        "Oracle TTC: 2.40 s",
        "Speed limit: 13.00 m/s",
        "Target speed: 8.00 m/s",
        "Shield: INTERVENED [minimum_ttc] (mode=enforce)",
        "Agent nominal: severity=0.20, recommended speed=12.00 m/s",
        "Agent hazard: severity=0.80, recommended speed=4.00 m/s",
        "Agent rule: severity=0.50, recommended speed=6.00 m/s",
        "Critic conflict: 0.65",
        "SMOKE - NOT A RESEARCH RESULT",
    )


def test_writes_gif_from_verified_lossless_frames_without_rerunning(
    step_bundle: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    _, trace, frames = step_bundle
    output = tmp_path / "episode.gif"

    write_episode_gif(trace, frames, output)

    rendered = list(iio.imiter(output))
    assert len(rendered) == 2
    assert all(frame.shape[0] >= 120 and frame.shape[1] == 220 for frame in rendered)
    assert any(frame.min() < 20 for frame in rendered)


def test_gif_is_repeatable_and_preserves_persisted_frame_identity_order(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    outputs: list[Path] = []
    decoded: list[list[np.ndarray]] = []
    for index in range(2):
        bundle = bundle_factory()
        trace = next(bundle.glob("episodes/**/*_trace.jsonl"))
        frames = trace.parent / "episode_20001_frames"
        output = tmp_path / f"repeat-{index}" / "episode.gif"
        write_episode_gif(trace, frames, output)
        outputs.append(output)
        decoded.append(list(iio.imiter(output)))

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert len(decoded[0]) == len(decoded[1]) == 2
    assert all(np.array_equal(left, right) for left, right in zip(*decoded, strict=True))
    assert tuple(int(frames[0][0, 0, 0]) for frames in decoded) == (60, 60)
    assert tuple(int(frames[1][0, 0, 0]) for frames in decoded) == (140, 140)


@pytest.mark.parametrize("artifact", ["jsonl", "frame"])
def test_gif_rejects_tampered_declared_inputs(
    step_bundle: tuple[Path, Path, Path], tmp_path: Path, artifact: str
) -> None:
    _, trace, frames = step_bundle
    if artifact == "jsonl":
        trace.write_bytes(trace.read_bytes() + b"tampered")
    else:
        next(frames.glob("*.png")).write_bytes(b"tampered")
    output = tmp_path / f"{artifact}.gif"
    with pytest.raises(ValueError, match="manifest|verification|integrity"):
        write_episode_gif(trace, frames, output)
    assert not output.exists()


@pytest.mark.parametrize("frame_path", ["../outside.png", "/outside.png", "C:/outside.png"])
def test_gif_rejects_absolute_and_traversal_record_frame_paths(
    step_bundle: tuple[Path, Path, Path], tmp_path: Path, frame_path: str
) -> None:
    bundle, trace, frames = step_bundle
    payloads = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    payloads[0]["frame_path"] = frame_path
    (bundle / "evaluation_manifest.json").unlink()
    trace.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in payloads),
        encoding="utf-8",
    )
    EvaluationWorkspace(destination=bundle, path=bundle).write_manifest()

    with pytest.raises(ValueError, match="frame_path|relative|outside|traversal"):
        write_episode_gif(trace, frames, tmp_path / "unsafe.gif")


def test_gif_rejects_declared_frame_outside_frames_directory(
    step_bundle: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    bundle, trace, frames = step_bundle
    payloads = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    payloads[0]["frame_path"] = "metrics/eval_metrics.csv"
    (bundle / "evaluation_manifest.json").unlink()
    trace.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in payloads),
        encoding="utf-8",
    )
    EvaluationWorkspace(destination=bundle, path=bundle).write_manifest()

    with pytest.raises(ValueError, match="frames_dir|outside"):
        write_episode_gif(trace, frames, tmp_path / "outside.gif")


def test_gif_does_not_overwrite_existing_output(
    step_bundle: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    _, trace, frames = step_bundle
    output = tmp_path / "episode.gif"
    output.write_bytes(b"owned")
    with pytest.raises(FileExistsError):
        write_episode_gif(trace, frames, output)
    assert output.read_bytes() == b"owned"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.replace("/000000.png", "/./000000.png"),
        lambda value: value.replace("/000000.png", "//000000.png"),
        lambda value: value.replace("/", "\\"),
    ),
    ids=("dot", "duplicate-separator", "backslash"),
)
def test_gif_rejects_noncanonical_frame_path_spelling(
    step_bundle: tuple[Path, Path, Path],
    tmp_path: Path,
    mutate: object,
) -> None:
    bundle, trace, frames = step_bundle
    payloads = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    original = str(payloads[0]["frame_path"])
    payloads[0]["frame_path"] = mutate(original)  # type: ignore[operator]
    _rewrite_trace_and_manifest(bundle, trace, payloads)

    with pytest.raises(ValueError, match="canonical|POSIX|frame_path"):
        write_episode_gif(trace, frames, tmp_path / "noncanonical.gif")


def test_gif_rejects_swapped_step_frame_order(
    step_bundle: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    bundle, trace, frames = step_bundle
    payloads = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    payloads[0]["frame_path"], payloads[1]["frame_path"] = (
        payloads[1]["frame_path"],
        payloads[0]["frame_path"],
    )
    _rewrite_trace_and_manifest(bundle, trace, payloads)

    with pytest.raises(ValueError, match="step|frame|order|identity"):
        write_episode_gif(trace, frames, tmp_path / "swapped.gif")


def test_gif_rejects_extra_declared_frame(
    step_bundle: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    bundle, trace, frames = step_bundle
    (bundle / "evaluation_manifest.json").unlink()
    iio.imwrite(
        frames / "000002.png",
        np.full((120, 220, 3), 200, dtype=np.uint8),
        extension=".png",
    )
    EvaluationWorkspace(destination=bundle, path=bundle).write_manifest()

    with pytest.raises(ValueError, match="exact|inventory|extra|frame"):
        write_episode_gif(trace, frames, tmp_path / "extra.gif")


def test_gif_rejects_missing_declared_frame(
    step_bundle: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    bundle, trace, frames = step_bundle
    (bundle / "evaluation_manifest.json").unlink()
    (frames / "000001.png").unlink()
    EvaluationWorkspace(destination=bundle, path=bundle).write_manifest()

    with pytest.raises(ValueError, match="exact|inventory|missing|frame"):
        write_episode_gif(trace, frames, tmp_path / "missing.gif")


def test_gif_rejects_unrelated_unique_frame_names(
    step_bundle: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    bundle, trace, frames = step_bundle
    payloads = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    (bundle / "evaluation_manifest.json").unlink()
    for index, payload in enumerate(payloads):
        old = frames / f"{index:06d}.png"
        new = frames / f"unrelated-{index}.png"
        old.rename(new)
        payload["frame_path"] = new.relative_to(bundle).as_posix()
    trace.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in payloads),
        encoding="utf-8",
    )
    EvaluationWorkspace(destination=bundle, path=bundle).write_manifest()

    with pytest.raises(ValueError, match="step|frame|identity|inventory"):
        write_episode_gif(trace, frames, tmp_path / "unrelated.gif")
