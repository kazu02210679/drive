from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

from mad_driving.evaluation.selection import CheckpointCandidate
from mad_driving.evaluation.training_metrics import (
    REQUIRED_TENSORBOARD_TAGS,
    TRAIN_METRICS_CSV_COLUMNS,
    TensorBoardEventSource,
    extract_training_metrics,
    extract_training_metrics_from_event_sources,
    write_training_metrics_csv,
)


def _candidate(run_dir: Path) -> CheckpointCandidate:
    return CheckpointCandidate(
        path=run_dir / "checkpoints" / "final_model.zip",
        sha256="a" * 64,
        resolved_config_sha256="e" * 64,
        method_id="proposed",
        policy_seed=42,
        checkpoint_kind="final",
        curriculum_level=3,
        training_timestep=200,
    )


def _periodic_candidate(run_dir: Path) -> CheckpointCandidate:
    return CheckpointCandidate(
        path=run_dir / "checkpoints" / "ppo_checkpoint_100_steps.zip",
        sha256="b" * 64,
        resolved_config_sha256="e" * 64,
        method_id="proposed",
        policy_seed=42,
        checkpoint_kind="periodic",
        curriculum_level=2,
        training_timestep=100,
    )


def _trust_run(monkeypatch: pytest.MonkeyPatch, run_dir: Path) -> list[Path]:
    verified: list[Path] = []

    def verify(candidate_run_dir: Path) -> tuple[CheckpointCandidate, ...]:
        verified.append(candidate_run_dir)
        return (_candidate(run_dir),)

    monkeypatch.setattr(
        "mad_driving.evaluation.selection.discover_checkpoint_candidates",
        verify,
    )
    return verified


def _write_events(
    run_dir: Path,
    *,
    tags: tuple[str, ...] = REQUIRED_TENSORBOARD_TAGS,
    value: float = 2.5,
) -> None:
    writer = SummaryWriter(log_dir=run_dir / "tensorboard" / "ppo_1")
    for timestep in (20, 10):
        for tag in reversed(tags):
            writer.add_scalar(tag, value if tag != "train/entropy_loss" else -0.75, timestep)
    writer.close()


def test_extracts_verified_offline_scalars_with_original_tags_and_entropy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run-42"
    _write_events(run_dir)
    verified = _trust_run(monkeypatch, run_dir)

    points = extract_training_metrics((run_dir,), smoke=False)

    assert verified == [run_dir]
    assert len(points) == 2 * (len(REQUIRED_TENSORBOARD_TAGS) + 1)
    assert {point.run_id for point in points} == {"run-42"}
    assert {point.method_id for point in points} == {"proposed"}
    assert {point.policy_seed for point in points} == {42}
    assert tuple(point.timestep for point in points) == tuple(
        sorted(point.timestep for point in points)
    )
    assert any(point.metric == "train/entropy_loss" and point.value == -0.75 for point in points)
    assert any(point.metric == "policy_entropy" and point.value == 0.75 for point in points)
    assert all(point.value is not None for point in points)


def test_event_source_extraction_remains_bound_to_authenticated_copied_bytes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-42"
    _write_events(run_dir)
    tensorboard_dir = run_dir / "tensorboard"
    event_files = tuple(tensorboard_dir.rglob("events.out.tfevents.*"))
    sources = tuple(
        TensorBoardEventSource(
            run_id=run_dir.name,
            method_id="proposed",
            policy_seed=42,
            event_relative_path=event.relative_to(tensorboard_dir).as_posix(),
            payload=(payload := event.read_bytes()),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for event in event_files
    )
    for event in event_files:
        event.write_bytes(b"source drift after private copy\n")

    points = extract_training_metrics_from_event_sources(sources, smoke=False)

    assert any(point.metric == "rollout/ep_rew_mean" and point.value == 2.5 for point in points)
    assert any(point.metric == "policy_entropy" and point.value == 0.75 for point in points)


def test_formal_missing_tag_fails_and_unverified_run_is_rejected_before_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "missing"
    missing_tag = "reward/jerk_penalty"
    _write_events(
        run_dir, tags=tuple(tag for tag in REQUIRED_TENSORBOARD_TAGS if tag != missing_tag)
    )
    _trust_run(monkeypatch, run_dir)
    with pytest.raises(ValueError, match=r"missing.*reward/jerk_penalty"):
        extract_training_metrics((run_dir,), smoke=False)

    unverified = tmp_path / "unverified"
    _write_events(unverified)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="metadata|complete|training"):
        extract_training_metrics((unverified,), smoke=False)


@pytest.mark.parametrize(
    "candidates",
    [
        pytest.param((_periodic_candidate,), id="no-final"),
        pytest.param((_candidate, _candidate), id="duplicate-final"),
    ],
)
def test_requires_exactly_one_authenticated_final_model_before_event_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidates: tuple[Callable[[Path], CheckpointCandidate], ...],
) -> None:
    run_dir = tmp_path / "checkpoint-inventory"

    def discover(candidate_run_dir: Path) -> tuple[CheckpointCandidate, ...]:
        assert candidate_run_dir == run_dir
        return tuple(factory(run_dir) for factory in candidates)

    monkeypatch.setattr(
        "mad_driving.evaluation.selection.discover_checkpoint_candidates",
        discover,
    )

    with pytest.raises(ValueError, match=r"exactly one authenticated final_model\.zip"):
        extract_training_metrics((run_dir,), smoke=False)


def test_smoke_records_missing_tags_as_none_and_csv_uses_fixed_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "smoke-run"
    _write_events(run_dir, tags=("rollout/ep_rew_mean",))
    _trust_run(monkeypatch, run_dir)

    points = extract_training_metrics((run_dir,), smoke=True)
    missing = {point.metric: point for point in points if point.value is None}

    assert set(REQUIRED_TENSORBOARD_TAGS) - {"rollout/ep_rew_mean"} <= set(missing)
    assert missing["policy_entropy"].timestep == 0
    destination = tmp_path / "metrics" / "train_metrics.csv"
    write_training_metrics_csv(destination, points, smoke=True)
    with destination.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert tuple(rows[0]) == TRAIN_METRICS_CSV_COLUMNS
    unavailable = next(row for row in rows if row["metric"] == "train/value_loss")
    assert unavailable["value"] == ""
    assert unavailable["result_label"] == "SMOKE - NOT A RESEARCH RESULT"
    assert destination.read_bytes().endswith(b"\n")
    assert b"\r\n" not in destination.read_bytes()


def test_nonfinite_event_scalar_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "nonfinite"
    _write_events(run_dir, value=float("nan"))
    _trust_run(monkeypatch, run_dir)

    with pytest.raises(ValueError, match="finite"):
        extract_training_metrics((run_dir,), smoke=False)


def test_metadata_verification_does_not_import_training_or_simulator_frameworks() -> None:
    script = r"""
import sys
import tempfile
from pathlib import Path
from mad_driving.evaluation.training_metrics import extract_training_metrics

run_dir = Path(tempfile.mkdtemp()) / "unverified"
run_dir.mkdir()
try:
    extract_training_metrics((run_dir,), smoke=False)
except ValueError:
    pass
for forbidden in ("metadrive", "stable_baselines3"):
    assert not any(name == forbidden or name.startswith(forbidden + ".") for name in sys.modules)
"""

    subprocess.run([sys.executable, "-c", script], check=True)
