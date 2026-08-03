"""Environment boundary types used by the Phase 1 smoke runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from mad_driving.interfaces import (
    CriticReview,
    DecisionTrace,
    RiskClaim,
    SceneSnapshot,
)


class DrivingEnvironment(Protocol):
    """Small subset of the MetaDrive environment API required by Phase 1."""

    config: dict[str, Any]
    vehicle: Any
    engine: Any
    agent: Any
    action_space: Any

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]: ...

    def step(
        self, action: tuple[float, float] | int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...


class EnvironmentFactory(Protocol):
    def __call__(self, config: dict[str, object]) -> DrivingEnvironment: ...


@dataclass(frozen=True)
class SmokeResult:
    """Finite summary of one fixed-action headless episode."""

    steps_completed: int
    terminated: bool
    truncated: bool
    final_snapshot: SceneSnapshot
    final_claims: tuple[RiskClaim, ...]
    final_review: CriticReview


@dataclass(frozen=True)
class ControlSmokeResult:
    """Finite summary of one shielded four-action control episode."""

    steps_completed: int
    terminated: bool
    truncated: bool
    final_snapshot: SceneSnapshot
    final_claims: tuple[RiskClaim, ...]
    final_review: CriticReview
    final_trace: DecisionTrace
    action_counts: tuple[int, int, int, int]
    shield_intervention_count: int


def create_metadrive_env(config: dict[str, object]) -> DrivingEnvironment:
    """Construct MetaDrive lazily so unit tests stay simulator-independent."""

    from metadrive import MetaDriveEnv  # type: ignore[import-untyped]

    return cast(DrivingEnvironment, MetaDriveEnv(config))
