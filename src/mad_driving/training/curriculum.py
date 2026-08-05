"""Immutable validation-driven Phase 5 curriculum state machine."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import yaml

from mad_driving.config.models import CurriculumConfig
from mad_driving.config.parsing import load_unique_yaml
from mad_driving.scenarios import EnvironmentRole

MAX_DIFFICULTY_LEVEL = 3
CURRICULUM_STATE_FILENAME = "curriculum_state.yaml"
CHECKPOINT_CURRICULUM_SIDECAR_SCHEMA_VERSION = 1
_CURRICULUM_STATE_FIELDS = frozenset({"level", "consecutive_passes", "evaluations"})
_CHECKPOINT_CURRICULUM_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_sha256",
        "level",
        "consecutive_passes",
        "evaluations",
    }
)
_EXPECTED_SCENARIOS_BY_LEVEL: dict[int, frozenset[str]] = {
    0: frozenset({"nominal"}),
    1: frozenset({"lead_brake"}),
    2: frozenset({"lead_brake", "cut_in"}),
    3: frozenset({"occluded_crossing"}),
}


def _strict_non_negative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative non-bool integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


@dataclass(frozen=True)
class CurriculumEpisodeResult:
    """Typed terminal metrics for one curriculum validation episode."""

    scenario_id: str
    success: bool
    collision: bool
    route_progress: float
    unnecessary_stop_duration_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario_id must be a non-empty string")
        if type(self.success) is not bool:
            raise ValueError("success must be a boolean")
        if type(self.collision) is not bool:
            raise ValueError("collision must be a boolean")
        if type(self.route_progress) is not float or not math.isfinite(self.route_progress):
            raise ValueError("route_progress must be a finite float")
        if not 0.0 <= self.route_progress <= 1.0:
            raise ValueError("route_progress must be from 0 through 1")
        if (
            type(self.unnecessary_stop_duration_s) is not float
            or not math.isfinite(self.unnecessary_stop_duration_s)
            or self.unnecessary_stop_duration_s < 0.0
        ):
            raise ValueError("unnecessary_stop_duration_s must be a non-negative finite float")


@dataclass(frozen=True)
class CurriculumState:
    """One validated, immutable curriculum snapshot."""

    level: int
    consecutive_passes: int
    evaluations: int

    def __post_init__(self) -> None:
        level = _strict_non_negative_integer("level", self.level)
        consecutive_passes = _strict_non_negative_integer(
            "consecutive_passes", self.consecutive_passes
        )
        evaluations = _strict_non_negative_integer("evaluations", self.evaluations)
        if level > MAX_DIFFICULTY_LEVEL:
            raise ValueError(f"level must be from 0 through {MAX_DIFFICULTY_LEVEL}")
        if consecutive_passes > evaluations:
            raise ValueError("consecutive_passes must not exceed evaluations")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "consecutive_passes", consecutive_passes)
        object.__setattr__(self, "evaluations", evaluations)


class CurriculumController:
    """Advance at most one difficulty level from scheduled validation results."""

    def __init__(self, config: CurriculumConfig, state: CurriculumState) -> None:
        if not isinstance(config, CurriculumConfig):
            raise TypeError("config must be a CurriculumConfig")
        if not isinstance(state, CurriculumState):
            raise TypeError("state must be a CurriculumState")
        if config.mode == "fixed" and state.level != config.fixed_level:
            raise ValueError("fixed curriculum state level must match fixed_level")
        if config.mode == "fixed" and state.consecutive_passes != 0:
            raise ValueError("fixed curriculum pass streak must be zero")
        if config.mode == "automatic" and state.level < config.initial_level:
            raise ValueError("automatic curriculum state cannot be below initial_level")
        if (
            config.mode == "automatic"
            and state.level < MAX_DIFFICULTY_LEVEL
            and state.consecutive_passes >= config.consecutive_evaluations
        ):
            raise ValueError(
                "automatic curriculum pass streak below level 3 must be less than "
                "consecutive_evaluations"
            )
        if config.mode == "automatic":
            minimum_evaluations = (
                state.level - config.initial_level
            ) * config.consecutive_evaluations + state.consecutive_passes
            if state.evaluations < minimum_evaluations:
                raise ValueError(
                    "automatic curriculum evaluations are insufficient for the stored "
                    "level and pass streak"
                )
        self._config = config
        self._state = state

    @property
    def state(self) -> CurriculumState:
        """Return the current immutable state."""

        return self._state

    def observe(
        self,
        role: EnvironmentRole,
        results: Sequence[CurriculumEpisodeResult],
    ) -> CurriculumState:
        """Observe one complete scheduled validation and return the updated state."""

        if role == "test":
            raise ValueError("test role cannot drive curriculum")
        if role != "validation":
            raise ValueError("curriculum observations must use the validation role")
        if isinstance(results, str | bytes) or not isinstance(results, Sequence):
            raise TypeError("results must be a sequence of CurriculumEpisodeResult values")
        episode_results = tuple(results)
        if not episode_results:
            raise ValueError("results must not be empty")
        if not all(isinstance(result, CurriculumEpisodeResult) for result in episode_results):
            raise TypeError("results must contain only CurriculumEpisodeResult values")

        evaluations = self._state.evaluations + 1
        if self._config.mode == "fixed":
            self._state = CurriculumState(self._state.level, 0, evaluations)
            return self._state

        expected_scenarios = _EXPECTED_SCENARIOS_BY_LEVEL[self._state.level]
        observed_scenarios = {result.scenario_id for result in episode_results}
        unexpected_scenarios = observed_scenarios - expected_scenarios
        if unexpected_scenarios:
            raise ValueError(
                "results contain scenarios outside the current difficulty level: "
                f"{sorted(unexpected_scenarios)!r}"
            )
        collision_rate = sum(result.collision for result in episode_results) / len(episode_results)
        scenario_success_rates: dict[str, float] = {}
        for scenario_id in expected_scenarios:
            scenario_results = tuple(
                result for result in episode_results if result.scenario_id == scenario_id
            )
            if not scenario_results:
                scenario_success_rates[scenario_id] = 0.0
                continue
            qualified_successes = sum(
                result.success
                and result.unnecessary_stop_duration_s
                <= self._config.maximum_unnecessary_stop_duration_s
                for result in scenario_results
            )
            scenario_success_rates[scenario_id] = qualified_successes / len(scenario_results)
        passed = (
            all(
                success_rate >= self._config.success_rate_threshold
                for success_rate in scenario_success_rates.values()
            )
            and collision_rate <= self._config.collision_rate_threshold
        )
        consecutive_passes = self._state.consecutive_passes + 1 if passed else 0
        should_advance = (
            consecutive_passes >= self._config.consecutive_evaluations
            and self._state.level < MAX_DIFFICULTY_LEVEL
        )
        level = self._state.level + 1 if should_advance else self._state.level
        if should_advance:
            consecutive_passes = 0
        self._state = CurriculumState(level, consecutive_passes, evaluations)
        return self._state


def write_curriculum_state(state: CurriculumState, destination: Path) -> None:
    """Atomically replace a curriculum snapshot through a durable sibling temp file."""

    if not isinstance(state, CurriculumState):
        raise TypeError("state must be a CurriculumState")
    destination_path = Path(destination)
    payload = {
        "level": state.level,
        "consecutive_passes": state.consecutive_passes,
        "evaluations": state.evaluations,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    primary_error: BaseException | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            yaml.safe_dump(payload, output, allow_unicode=True, sort_keys=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination_path)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"Curriculum state temporary cleanup also failed: {cleanup_error}"
                )


def checkpoint_curriculum_sidecar_path(checkpoint: str | Path) -> Path:
    """Return the adjacent state sidecar bound to one checkpoint."""

    checkpoint_path = Path(checkpoint)
    return checkpoint_path.with_name(f"{checkpoint_path.name}.curriculum.yaml")


def _sha256_path(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_checkpoint_curriculum_state(
    state: CurriculumState,
    checkpoint: str | Path,
) -> Path:
    """Atomically bind one immutable curriculum snapshot to a saved checkpoint."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found for curriculum binding: {checkpoint_path}")
    destination = checkpoint_curriculum_sidecar_path(checkpoint_path)
    payload = {
        "schema_version": CHECKPOINT_CURRICULUM_SIDECAR_SCHEMA_VERSION,
        "checkpoint_sha256": _sha256_path(checkpoint_path),
        "level": state.level,
        "consecutive_passes": state.consecutive_passes,
        "evaluations": state.evaluations,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    primary_error: BaseException | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            yaml.safe_dump(payload, output, allow_unicode=True, sort_keys=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"Checkpoint curriculum temporary cleanup also failed: {cleanup_error}"
                )
    return destination


def read_stable_artifact_bytes(
    source_path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[bytes, str]:
    descriptor = os.open(
        source_path,
        os.O_RDONLY | getattr(os, "O_BINARY", 0),
    )
    try:
        with os.fdopen(descriptor, "rb") as source:
            before = os.fstat(source.fileno())
            data = source.read()
            path_stat = source_path.stat()
            after = os.fstat(source.fileno())
            stable_fields = ("st_size", "st_mtime_ns", "st_ctime_ns")
            if (
                not os.path.samestat(before, after)
                or not os.path.samestat(after, path_stat)
                or any(getattr(before, field) != getattr(after, field) for field in stable_fields)
                or len(data) != after.st_size
            ):
                raise ValueError(f"Artifact changed or was replaced while reading: {source_path}")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Artifact SHA-256 mismatch: {source_path}")
    return data, digest


def read_checkpoint_curriculum_artifact(
    sidecar: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_sha256: str | None = None,
) -> tuple[CurriculumState, str]:
    """Read and hash one stable checkpoint-bound curriculum byte snapshot."""

    source_path = Path(sidecar)
    try:
        data, digest = read_stable_artifact_bytes(
            source_path,
            expected_sha256=expected_sha256,
        )
        payload = load_unique_yaml(data.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(
            f"Checkpoint curriculum state is malformed: {source_path}: {error}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_CURRICULUM_FIELDS:
        raise ValueError(f"Checkpoint curriculum state fields are malformed: {source_path}")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != CHECKPOINT_CURRICULUM_SIDECAR_SCHEMA_VERSION
    ):
        raise ValueError(f"Checkpoint curriculum schema is malformed: {source_path}")
    if payload["checkpoint_sha256"] != expected_checkpoint_sha256:
        raise ValueError("Checkpoint curriculum state is bound to a different checkpoint")
    try:
        state = CurriculumState(
            level=payload["level"],
            consecutive_passes=payload["consecutive_passes"],
            evaluations=payload["evaluations"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Checkpoint curriculum state values are malformed: {source_path}"
        ) from error
    return state, digest


def read_checkpoint_curriculum_state(
    sidecar: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_sha256: str | None = None,
) -> CurriculumState:
    """Return the state parsed from one stable checkpoint-bound byte snapshot."""

    state, _digest = read_checkpoint_curriculum_artifact(
        sidecar,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_sha256=expected_sha256,
    )
    return state


def read_curriculum_state_artifact(
    source: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[CurriculumState, str]:
    """Read, authenticate, and parse one stable curriculum byte snapshot."""

    source_path = Path(source)
    try:
        data, digest = read_stable_artifact_bytes(
            source_path,
            expected_sha256=expected_sha256,
        )
        payload = load_unique_yaml(data.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Curriculum state is malformed: {source_path}: {error}") from error
    if not isinstance(payload, dict) or set(payload) != _CURRICULUM_STATE_FIELDS:
        raise ValueError(f"Curriculum state fields are malformed: {source_path}")
    try:
        state = CurriculumState(
            level=payload["level"],
            consecutive_passes=payload["consecutive_passes"],
            evaluations=payload["evaluations"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Curriculum state values are malformed: {source_path}") from error
    return state, digest


def read_curriculum_state(source: Path) -> CurriculumState:
    """Strictly parse one persisted curriculum snapshot."""

    state, _digest = read_curriculum_state_artifact(source)
    return state


__all__ = [
    "CHECKPOINT_CURRICULUM_SIDECAR_SCHEMA_VERSION",
    "CURRICULUM_STATE_FILENAME",
    "MAX_DIFFICULTY_LEVEL",
    "CurriculumController",
    "CurriculumEpisodeResult",
    "CurriculumState",
    "checkpoint_curriculum_sidecar_path",
    "read_checkpoint_curriculum_artifact",
    "read_checkpoint_curriculum_state",
    "read_curriculum_state",
    "read_curriculum_state_artifact",
    "read_stable_artifact_bytes",
    "write_checkpoint_curriculum_state",
    "write_curriculum_state",
]
