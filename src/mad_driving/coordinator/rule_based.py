"""Deterministic baseline Coordinator for structured agent claims."""

from collections.abc import Sequence

from mad_driving.config.models import CoordinatorConfig
from mad_driving.control import DrivingAction, action_for_speed_cap
from mad_driving.interfaces import CriticReview, RiskClaim, SceneSnapshot
from mad_driving.interfaces.defensive_validation import (
    valid_claim,
    valid_review,
    valid_snapshot,
)


class RuleBasedCoordinator:
    """Aggregate claim recommendations without duplicating Shield physics."""

    _required_agent_ids = frozenset({"nominal", "hazard", "rule"})

    def __init__(self, config: CoordinatorConfig) -> None:
        self._config = config

    def decide(
        self,
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> DrivingAction:
        """Return the most restrictive configured candidate action."""

        if (
            not valid_snapshot(snapshot)
            or not valid_review(review)
            or any(not valid_claim(claim) for claim in claims)
        ):
            return DrivingAction.PREPARE_STOP

        present_agent_ids = {claim.agent_id for claim in claims}
        if self._required_agent_ids - present_agent_ids:
            base = DrivingAction.PREPARE_STOP
        else:
            base = max(
                action_for_speed_cap(
                    claim.recommended_max_speed_mps,
                    snapshot.ego.speed_limit_mps,
                )
                for claim in claims
            )

        candidates = [base]
        if any(claim.hard_stop_required for claim in claims):
            candidates.append(DrivingAction.STOP)
        if review.unresolved_conflict:
            candidates.append(DrivingAction(self._config.conflict_min_action))
        if review.max_severity >= self._config.severe_threshold:
            candidates.append(DrivingAction(self._config.severe_min_action))
        return max(candidates)
