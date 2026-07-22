from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import pytest

from mad_driving.evaluation.workspace import EvaluationWorkspace
from mad_driving.visualization.overlay import _overlay_lines, write_episode_gif

from .conftest import make_step


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
