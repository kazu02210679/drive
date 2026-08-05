"""Offline extraction of authenticated completed-run TensorBoard scalars."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from mad_driving.evaluation.models import REWARD_COMPONENT_KEYS

SMOKE_RESULT_LABEL: Final = "SMOKE - NOT A RESEARCH RESULT"
REQUIRED_TENSORBOARD_TAGS: Final = (
    "rollout/ep_rew_mean",
    *(f"reward/{name}" for name in REWARD_COMPONENT_KEYS),
    "train/entropy_loss",
    "train/value_loss",
    "train/explained_variance",
)
_METRIC_ORDER: Final = {
    metric: index
    for index, metric in enumerate(
        (
            *REQUIRED_TENSORBOARD_TAGS,
            "policy_entropy",
        )
    )
}
_EVENT_FILE_PREFIX: Final = "events.out.tfevents."
_FILE_ATTRIBUTE_REPARSE_POINT: Final = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

TRAIN_METRICS_CSV_COLUMNS: Final = (
    "result_label",
    "run_id",
    "method_id",
    "policy_seed",
    "timestep",
    "metric",
    "value",
)


@dataclass(frozen=True)
class TrainingMetricPoint:
    run_id: str
    method_id: str
    policy_seed: int
    timestep: int
    metric: str
    value: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("method_id must be a non-empty string")
        if type(self.policy_seed) is not int or self.policy_seed < 0:
            raise ValueError("policy_seed must be a non-negative integer")
        if type(self.timestep) is not int or self.timestep < 0:
            raise ValueError("timestep must be a non-negative integer")
        if self.metric not in _METRIC_ORDER:
            raise ValueError("metric is not a supported training metric")
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, int | float):
                raise TypeError("value must be numeric when available")
            numeric = float(self.value)
            if not math.isfinite(numeric):
                raise ValueError("value must be finite when available")
            object.__setattr__(self, "value", numeric)


@dataclass(frozen=True)
class TensorBoardEventSource:
    """Authenticated immutable TensorBoard bytes copied into the source snapshot."""

    run_id: str
    method_id: str
    policy_seed: int
    event_relative_path: str
    payload: bytes
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("method_id must be a non-empty string")
        if type(self.policy_seed) is not int or self.policy_seed < 0:
            raise ValueError("policy_seed must be a non-negative integer")
        if not isinstance(self.event_relative_path, str) or not self.event_relative_path:
            raise ValueError("event_relative_path must be a non-empty string")
        relative = PurePosixPath(self.event_relative_path)
        if (
            relative.is_absolute()
            or relative.as_posix() != self.event_relative_path
            or "\\" in self.event_relative_path
            or any(part in {"", ".", ".."} for part in relative.parts)
            or not relative.name.startswith(_EVENT_FILE_PREFIX)
        ):
            raise ValueError("event_relative_path must identify a relative TensorBoard event")
        if type(self.payload) is not bytes:
            raise TypeError("payload must be bytes")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if self.sha256 != self.sha256.lower() or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError("payload does not match its SHA-256 binding")


def _is_reparse(entry_stat: os.stat_result) -> bool:
    attributes = getattr(entry_stat, "st_file_attributes", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _event_files(tensorboard_dir: Path) -> tuple[Path, ...]:
    try:
        root_stat = tensorboard_dir.lstat()
    except OSError as error:
        raise ValueError(f"TensorBoard directory is unavailable: {tensorboard_dir}") from error
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or tensorboard_dir.is_symlink()
        or _is_reparse(root_stat)
    ):
        raise ValueError(f"TensorBoard path is not a tracked regular directory: {tensorboard_dir}")

    event_files: list[Path] = []
    pending = [tensorboard_dir]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError(f"TensorBoard directory is unreadable: {directory}") from error
        child_dirs: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"TensorBoard inventory entry is unreadable: {path}") from error
            if entry.is_symlink() or _is_reparse(entry_stat):
                raise ValueError(
                    f"TensorBoard inventory contains a symlink or reparse entry: {path}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                child_dirs.append(path)
            elif stat.S_ISREG(entry_stat.st_mode):
                if entry.name.startswith(_EVENT_FILE_PREFIX):
                    event_files.append(path)
            else:
                raise ValueError(f"TensorBoard inventory contains a special entry: {path}")
        pending.extend(reversed(child_dirs))
    if not event_files:
        raise ValueError(f"TensorBoard directory contains no event files: {tensorboard_dir}")
    return tuple(sorted(event_files, key=lambda path: path.relative_to(tensorboard_dir).as_posix()))


def _event_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        identity = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"TensorBoard event file is unreadable: {path}") from error
    if not stat.S_ISREG(identity.st_mode) or path.is_symlink() or _is_reparse(identity):
        raise ValueError(f"TensorBoard event path is not a regular file: {path}")
    return (identity.st_dev, identity.st_ino, identity.st_size, identity.st_mtime_ns)


def _verified_run_provenance(run_dir: Path) -> tuple[str, int]:
    from mad_driving.evaluation.selection import discover_checkpoint_candidates

    candidates = discover_checkpoint_candidates(run_dir)
    final_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.checkpoint_kind == "final" and candidate.path.name == "final_model.zip"
    )
    if len(final_candidates) != 1:
        raise ValueError(
            "completed training run must have exactly one authenticated "
            f"final_model.zip candidate: {run_dir}"
        )
    provenances = {(candidate.method_id, candidate.policy_seed) for candidate in candidates}
    if len(provenances) != 1:
        raise ValueError(f"completed training run has inconsistent provenance: {run_dir}")
    final_candidate = final_candidates[0]
    return final_candidate.method_id, final_candidate.policy_seed


def _read_event_points(
    event_files: Sequence[Path],
    *,
    run_id: str,
    method_id: str,
    policy_seed: int,
    smoke: bool,
) -> list[TrainingMetricPoint]:
    from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore[import-untyped]
        EventAccumulator,
    )

    scalars: dict[str, list[tuple[int, float]]] = {tag: [] for tag in REQUIRED_TENSORBOARD_TAGS}
    seen: set[tuple[str, int]] = set()
    for event_file in event_files:
        identity = _event_identity(event_file)
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            scalar_tags = set(accumulator.Tags().get("scalars", ()))
            for tag in REQUIRED_TENSORBOARD_TAGS:
                if tag not in scalar_tags:
                    continue
                for event in accumulator.Scalars(tag):
                    timestep = int(event.step)
                    value = float(event.value)
                    if timestep < 0:
                        raise ValueError(
                            f"TensorBoard scalar has a negative timestep in run {run_id}"
                        )
                    if not math.isfinite(value):
                        raise ValueError(
                            f"TensorBoard scalar must be finite in run {run_id}: {tag}"
                        )
                    event_key = (tag, timestep)
                    if event_key in seen:
                        raise ValueError(
                            f"TensorBoard scalar is duplicated in run {run_id}: {tag}@{timestep}"
                        )
                    seen.add(event_key)
                    scalars[tag].append((timestep, value))
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"TensorBoard event file is malformed: {event_file}") from error
        if _event_identity(event_file) != identity:
            raise ValueError(f"TensorBoard event file changed while being read: {event_file}")

    missing = tuple(tag for tag in REQUIRED_TENSORBOARD_TAGS if not scalars[tag])
    if missing and not smoke:
        raise ValueError(f"training run {run_id} is missing required tags: {', '.join(missing)}")

    points: list[TrainingMetricPoint] = []
    for tag in REQUIRED_TENSORBOARD_TAGS:
        values = sorted(scalars[tag])
        if not values:
            points.append(TrainingMetricPoint(run_id, method_id, policy_seed, 0, tag, None))
            if tag == "train/entropy_loss":
                points.append(
                    TrainingMetricPoint(
                        run_id,
                        method_id,
                        policy_seed,
                        0,
                        "policy_entropy",
                        None,
                    )
                )
            continue
        for timestep, value in values:
            points.append(TrainingMetricPoint(run_id, method_id, policy_seed, timestep, tag, value))
            if tag == "train/entropy_loss":
                points.append(
                    TrainingMetricPoint(
                        run_id,
                        method_id,
                        policy_seed,
                        timestep,
                        "policy_entropy",
                        -value,
                    )
                )
    return points


def _read_run_points(
    run_dir: Path,
    *,
    run_id: str,
    method_id: str,
    policy_seed: int,
    smoke: bool,
) -> list[TrainingMetricPoint]:
    return _read_event_points(
        _event_files(run_dir / "tensorboard"),
        run_id=run_id,
        method_id=method_id,
        policy_seed=policy_seed,
        smoke=smoke,
    )


def extract_training_metrics(
    run_dirs: Sequence[Path],
    *,
    smoke: bool,
) -> tuple[TrainingMetricPoint, ...]:
    """Verify completed runs, then read their TensorBoard events without training."""

    if type(smoke) is not bool:
        raise TypeError("smoke must be a bool")
    directories = tuple(Path(run_dir) for run_dir in run_dirs)
    if not directories:
        raise ValueError("run_dirs must be non-empty")
    ordered = sorted(directories, key=lambda path: (path.name, str(path)))
    run_ids = tuple(path.name for path in ordered)
    if any(not run_id for run_id in run_ids) or len(run_ids) != len(set(run_ids)):
        raise ValueError("training run directory names must provide unique run_id values")

    points: list[TrainingMetricPoint] = []
    for run_dir, run_id in zip(ordered, run_ids, strict=True):
        method_id, policy_seed = _verified_run_provenance(run_dir)
        points.extend(
            _read_run_points(
                run_dir,
                run_id=run_id,
                method_id=method_id,
                policy_seed=policy_seed,
                smoke=smoke,
            )
        )
    return tuple(
        sorted(
            points,
            key=lambda point: (
                point.run_id,
                point.timestep,
                _METRIC_ORDER[point.metric],
            ),
        )
    )


def extract_training_metrics_from_event_sources(
    sources: Sequence[TensorBoardEventSource],
    *,
    smoke: bool,
) -> tuple[TrainingMetricPoint, ...]:
    """Read metrics solely from authenticated bytes in a private source snapshot."""

    if type(smoke) is not bool:
        raise TypeError("smoke must be a bool")
    values = tuple(sources)
    if not values:
        raise ValueError("sources must be non-empty")
    if any(type(source) is not TensorBoardEventSource for source in values):
        raise TypeError("sources must contain TensorBoardEventSource values")

    grouped: dict[tuple[str, str, int], list[TensorBoardEventSource]] = {}
    run_provenance: dict[str, tuple[str, int]] = {}
    for source in values:
        provenance = (source.method_id, source.policy_seed)
        previous = run_provenance.setdefault(source.run_id, provenance)
        if previous != provenance:
            raise ValueError("event sources disagree on authenticated run provenance")
        grouped.setdefault((source.run_id, *provenance), []).append(source)

    points: list[TrainingMetricPoint] = []
    with tempfile.TemporaryDirectory(prefix="mad-driving-tensorboard-") as temporary:
        temporary_root = Path(temporary)
        for run_index, ((run_id, method_id, policy_seed), run_sources) in enumerate(
            sorted(grouped.items())
        ):
            relative_paths = [source.event_relative_path for source in run_sources]
            if len(relative_paths) != len(set(relative_paths)):
                raise ValueError(f"event sources contain a duplicate path in run {run_id}")
            event_files: list[Path] = []
            for event_index, source in enumerate(
                sorted(run_sources, key=lambda item: item.event_relative_path)
            ):
                if hashlib.sha256(source.payload).hexdigest() != source.sha256:
                    raise ValueError("event source payload changed after authentication")
                destination = temporary_root.joinpath(
                    f"run-{run_index}", f"{_EVENT_FILE_PREFIX}{event_index}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as output:
                    output.write(source.payload)
                event_files.append(destination)
            points.extend(
                _read_event_points(
                    event_files,
                    run_id=run_id,
                    method_id=method_id,
                    policy_seed=policy_seed,
                    smoke=smoke,
                )
            )
    return tuple(
        sorted(
            points,
            key=lambda point: (
                point.run_id,
                point.timestep,
                _METRIC_ORDER[point.metric],
            ),
        )
    )


def write_training_metrics_csv(
    path: Path,
    points: Sequence[TrainingMetricPoint],
    *,
    smoke: bool,
) -> None:
    """Write fixed-column offline training metrics with explicit unavailable cells."""

    if type(smoke) is not bool:
        raise TypeError("smoke must be a bool")
    values = tuple(points)
    if any(type(point) is not TrainingMetricPoint for point in values):
        raise TypeError("points must contain TrainingMetricPoint values")
    if not smoke and any(point.value is None for point in values):
        raise ValueError("formal training metrics cannot contain unavailable values")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=TRAIN_METRICS_CSV_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for point in values:
            writer.writerow(
                {
                    "result_label": SMOKE_RESULT_LABEL if smoke else "",
                    "run_id": point.run_id,
                    "method_id": point.method_id,
                    "policy_seed": point.policy_seed,
                    "timestep": point.timestep,
                    "metric": point.metric,
                    "value": "" if point.value is None else point.value,
                }
            )


__all__ = [
    "REQUIRED_TENSORBOARD_TAGS",
    "TRAIN_METRICS_CSV_COLUMNS",
    "TensorBoardEventSource",
    "TrainingMetricPoint",
    "extract_training_metrics",
    "extract_training_metrics_from_event_sources",
    "write_training_metrics_csv",
]
