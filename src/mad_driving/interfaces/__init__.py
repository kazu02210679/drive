"""Immutable interfaces shared by driving components."""

from typing import TYPE_CHECKING

from mad_driving.interfaces.actor_state import ActorState
from mad_driving.interfaces.critic_review import CriticReview
from mad_driving.interfaces.decision_trace import DecisionTrace
from mad_driving.interfaces.risk_claim import RiskClaim
from mad_driving.interfaces.scene_frame import (
    CollisionKind,
    OcclusionRegion,
    PrivilegedWorldState,
    RoadContext,
    SceneFrame,
    stopping_margin_m,
)
from mad_driving.interfaces.scene_snapshot import EgoState, SceneObservation, SceneSnapshot

if TYPE_CHECKING:
    from mad_driving.interfaces.shield_result import ShieldResult


def __getattr__(name: str) -> object:
    """Load the control-dependent Shield result only when requested."""

    if name == "ShieldResult":
        from mad_driving.interfaces.shield_result import ShieldResult

        return ShieldResult
    raise AttributeError(name)


__all__ = [
    "ActorState",
    "CollisionKind",
    "CriticReview",
    "DecisionTrace",
    "EgoState",
    "OcclusionRegion",
    "PrivilegedWorldState",
    "RiskClaim",
    "RoadContext",
    "SceneFrame",
    "SceneObservation",
    "SceneSnapshot",
    "ShieldResult",
    "stopping_margin_m",
]
