"""Immutable interfaces shared by driving components."""

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
)
from mad_driving.interfaces.scene_snapshot import EgoState, SceneObservation, SceneSnapshot
from mad_driving.interfaces.shield_result import ShieldResult

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
]
