"""Defensive reconstruction checks for externally corrupted dataclasses."""

from dataclasses import asdict

from mad_driving.interfaces.actor_state import ActorState
from mad_driving.interfaces.critic_review import CriticReview
from mad_driving.interfaces.risk_claim import RiskClaim
from mad_driving.interfaces.scene_frame import OcclusionRegion, RoadContext
from mad_driving.interfaces.scene_snapshot import EgoState, SceneObservation


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


def valid_snapshot(snapshot: SceneObservation) -> bool:
    """Return whether a nested scene still satisfies all typed invariants."""

    try:
        values = asdict(snapshot)
        ego = EgoState(**values.pop("ego"))
        visible_actors = tuple(ActorState(**actor) for actor in values.pop("visible_actors"))
        occlusion_regions = tuple(
            OcclusionRegion(**region) for region in values.pop("occlusion_regions")
        )
        road_context = RoadContext(**values.pop("road_context"))
        SceneObservation(
            ego=ego,
            visible_actors=visible_actors,
            occlusion_regions=occlusion_regions,
            road_context=road_context,
            **values,
        )
    except (TypeError, ValueError):
        return False
    return True
