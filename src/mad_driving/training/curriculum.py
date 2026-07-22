"""Immutable validation-driven Phase 5 curriculum state machine."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import yaml

from mad_driving.config.models import CurriculumConfig
from mad_driving.scenarios import EnvironmentRole

MAX_DIFFICULTY_LEVEL = 3
CURRICULUM_STATE_FILENAME = "curriculum_state.yaml"
_CURRICULUM_STATE_FIELDS = frozenset({"level", "consecutive_passes", "evaluations"})


def _strict_non_negative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative non-bool integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


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
        if config.mode == "automatic" and state.level < config.initial_level:
            raise ValueError("automatic curriculum state cannot be below initial_level")
        self._config = config
        self._state = state

    @property
    def state(self) -> CurriculumState:
        """Return the current immutable state."""

        return self._state

    def observe(
        self,
        role: EnvironmentRole,
        successes: int,
        collisions: int,
        episodes: int,
    ) -> CurriculumState:
        """Observe one complete scheduled validation and return the updated state."""

        if role == "test":
            raise ValueError("test role cannot drive curriculum")
        if role != "validation":
            raise ValueError("curriculum observations must use the validation role")
        episode_count = _strict_non_negative_integer("episodes", episodes)
        if episode_count == 0:
            raise ValueError("episodes must be greater than zero")
        success_count = _strict_non_negative_integer("successes", successes)
        collision_count = _strict_non_negative_integer("collisions", collisions)
        if success_count > episode_count:
            raise ValueError("successes must not exceed episodes")
        if collision_count > episode_count:
            raise ValueError("collisions must not exceed episodes")

        evaluations = self._state.evaluations + 1
        if self._config.mode == "fixed":
            self._state = CurriculumState(self._state.level, 0, evaluations)
            return self._state

        passed = success_count / episode_count >= self._config.success_rate_threshold and (
            collision_count / episode_count <= self._config.collision_rate_threshold
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


def read_curriculum_state(source: Path) -> CurriculumState:
    """Strictly parse one persisted curriculum snapshot."""

    source_path = Path(source)
    try:
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Curriculum state is malformed: {source_path}") from error
    if not isinstance(payload, dict) or set(payload) != _CURRICULUM_STATE_FIELDS:
        raise ValueError(f"Curriculum state fields are malformed: {source_path}")
    try:
        return CurriculumState(
            level=payload["level"],
            consecutive_passes=payload["consecutive_passes"],
            evaluations=payload["evaluations"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Curriculum state values are malformed: {source_path}") from error


__all__ = [
    "CURRICULUM_STATE_FILENAME",
    "MAX_DIFFICULTY_LEVEL",
    "CurriculumController",
    "CurriculumState",
    "read_curriculum_state",
    "write_curriculum_state",
]
