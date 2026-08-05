"""Intentional neutral reviewer for method ablations."""

from collections.abc import Sequence

from mad_driving.interfaces import CriticReview, RiskClaim, SceneObservation


class NoOpCritic:
    """Return a neutral review when the Critic is intentionally disabled."""

    def review(
        self,
        observation: SceneObservation,
        claims: Sequence[RiskClaim],
        *,
        failed_agent_ids: Sequence[str],
    ) -> CriticReview:
        del observation, claims, failed_agent_ids
        return CriticReview(
            conflict_score=0.0,
            unresolved_conflict=False,
            max_severity=0.0,
            supported_agent_ids=(),
            challenged_claim_ids=(),
            reasons=("critic_intentionally_disabled",),
        )
