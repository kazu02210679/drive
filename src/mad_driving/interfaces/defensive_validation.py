"""Defensive reconstruction checks for externally corrupted dataclasses."""

from dataclasses import asdict

from mad_driving.interfaces.actor_state import ActorState
from mad_driving.interfaces.critic_review import CriticReview
from mad_driving.interfaces.risk_claim import RiskClaim
from mad_driving.interfaces.scene_snapshot import EgoState, SceneSnapshot


def valid_claim(claim: RiskClaim) -> bool:
    """Return whether a claim still satisfies its constructor invariants."""

    try:
        RiskClaim(**asdict(claim))
    except (TypeError, ValueError):
        return False
    return True


def valid_review(review: CriticReview) -> bool:
    """Return whether a review still satisfies its constructor invariants."""

    try:
        CriticReview(**asdict(review))
    except (TypeError, ValueError):
        return False
    return True


def valid_snapshot(snapshot: SceneSnapshot) -> bool:
    """Return whether a nested scene still satisfies all typed invariants."""

    try:
        values = asdict(snapshot)
        ego = EgoState(**values.pop("ego"))
        actors = tuple(ActorState(**actor) for actor in values.pop("actors"))
        SceneSnapshot(ego=ego, actors=actors, **values)
    except (TypeError, ValueError):
        return False
    return True
