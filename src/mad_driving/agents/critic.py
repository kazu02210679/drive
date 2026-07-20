"""Defensive one-pass cross-review for deterministic risk claims."""

from collections.abc import Sequence

from mad_driving.config.models import CriticAgentConfig
from mad_driving.interfaces import CriticReview, RiskClaim, SceneSnapshot


class CriticAgent:
    """Apply the fixed Phase 2 review rules without rerunning any agent."""

    _rule_count = 8

    def __init__(self, config: CriticAgentConfig) -> None:
        self._config = config

    def review(self, snapshot: SceneSnapshot, claims: Sequence[RiskClaim]) -> CriticReview:
        indexed_claims = tuple(enumerate(claims))
        valid: list[tuple[int, RiskClaim]] = []
        invalid: list[tuple[int, RiskClaim]] = []
        for indexed_claim in indexed_claims:
            destination = valid if self._is_valid(indexed_claim[1]) else invalid
            destination.append(indexed_claim)

        reasons: list[str] = []
        challenged_indexes: set[int] = set()

        nominal_low = [
            item
            for item in valid
            if item[1].agent_id == "nominal"
            and item[1].severity < self._config.nominal_low_risk_threshold
        ]
        hazard_negative = [
            item
            for item in valid
            if item[1].agent_id == "hazard"
            and item[1].stopping_margin_m is not None
            and item[1].stopping_margin_m < 0.0
        ]
        if nominal_low and hazard_negative:
            reasons.append("nominal_hazard_disagreement")
            self._challenge(challenged_indexes, nominal_low, hazard_negative)

        if snapshot.occlusion_present and nominal_low:
            reasons.append("occlusion_underestimated")
            self._challenge(challenged_indexes, nominal_low)

        rule_stops = [
            item for item in valid if item[1].agent_id == "rule" and item[1].hard_stop_required
        ]
        moving_others = [
            item
            for item in valid
            if item[1].agent_id != "rule" and item[1].recommended_max_speed_mps > 0.0
        ]
        if rule_stops and moving_others:
            reasons.append("hard_stop_conflict")
            self._challenge(challenged_indexes, rule_stops, moving_others)

        if valid:
            recommendations = [claim.recommended_max_speed_mps for _, claim in valid]
            if max(recommendations) - min(recommendations) > self._config.recommendation_spread_mps:
                reasons.append("speed_recommendation_spread")
                self._challenge(challenged_indexes, valid)

        expired = [item for item in valid if item[1].valid_until_step < snapshot.step_index]
        if expired:
            reasons.append("claim_expired")
            self._challenge(challenged_indexes, expired)

        definitive_limit_mps = snapshot.ego.speed_limit_mps * self._config.definitive_speed_fraction
        low_confidence_definitive = [
            item
            for item in valid
            if item[1].confidence < self._config.low_confidence_threshold
            and (
                item[1].hard_stop_required
                or item[1].recommended_max_speed_mps <= definitive_limit_mps
            )
        ]
        if low_confidence_definitive:
            reasons.append("low_confidence_definitive")
            self._challenge(challenged_indexes, low_confidence_definitive)

        missing_evidence = [item for item in valid if not item[1].evidence]
        if missing_evidence:
            reasons.append("missing_evidence")
            self._challenge(challenged_indexes, missing_evidence)

        if invalid:
            reasons.append("invalid_claim")
            self._challenge(challenged_indexes, invalid)

        challenged_claim_ids = self._challenged_ids(indexed_claims, challenged_indexes)
        supported_agent_ids = tuple(
            sorted({claim.agent_id for index, claim in valid if index not in challenged_indexes})
        )
        max_severity = 1.0 if invalid else max((claim.severity for _, claim in valid), default=0.0)
        return CriticReview(
            conflict_score=len(reasons) / self._rule_count,
            unresolved_conflict=bool(reasons),
            max_severity=max_severity,
            supported_agent_ids=supported_agent_ids,
            challenged_claim_ids=challenged_claim_ids,
            reasons=tuple(reasons),
        )

    @staticmethod
    def _challenge(indexes: set[int], *groups: Sequence[tuple[int, RiskClaim]]) -> None:
        for group in groups:
            indexes.update(index for index, _ in group)

    @staticmethod
    def _challenged_ids(
        indexed_claims: Sequence[tuple[int, RiskClaim]], challenged_indexes: set[int]
    ) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for index, claim in indexed_claims:
            if index in challenged_indexes and claim.claim_id not in seen:
                seen.add(claim.claim_id)
                result.append(claim.claim_id)
        return tuple(result)

    @staticmethod
    def _is_valid(claim: RiskClaim) -> bool:
        try:
            RiskClaim(
                claim_id=claim.claim_id,
                agent_id=claim.agent_id,
                event_type=claim.event_type,
                target_actor_id=claim.target_actor_id,
                probability=claim.probability,
                confidence=claim.confidence,
                severity=claim.severity,
                time_horizon_s=claim.time_horizon_s,
                min_ttc_s=claim.min_ttc_s,
                stopping_margin_m=claim.stopping_margin_m,
                recommended_max_speed_mps=claim.recommended_max_speed_mps,
                hard_stop_required=claim.hard_stop_required,
                evidence=claim.evidence,
                assumptions=claim.assumptions,
                valid_until_step=claim.valid_until_step,
            )
        except (TypeError, ValueError):
            return False
        return True
