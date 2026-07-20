"""Fixed 24-dimensional coordinator observation assembly."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from mad_driving.config.models import ObservationConfig
from mad_driving.control import target_speed_mps
from mad_driving.interfaces import CriticReview, RiskClaim, SceneObservation
from mad_driving.interfaces.defensive_validation import valid_claim, valid_review, valid_snapshot

_REQUIRED_AGENT_IDS = ("nominal", "hazard", "rule")


@dataclass(frozen=True)
class _AggregatedClaim:
    """Safety-conservative feature values for one specialist's claims."""

    agent_id: str
    min_ttc_s: float | None
    stopping_margin_m: float | None
    probability: float | None
    confidence: float
    severity: float
    recommended_max_speed_mps: float
    hard_stop_required: bool


def aggregate_agent_claims(
    agent_id: str, claims: Sequence[RiskClaim]
) -> _AggregatedClaim | None:
    """Return one conservative aggregate, or ``None`` when an agent made no claim."""

    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("invalid agent_id")
    if any(not valid_claim(claim) for claim in claims):
        raise ValueError("invalid claim")

    agent_claims = tuple(claim for claim in claims if claim.agent_id == agent_id)
    if not agent_claims:
        return None

    finite_ttc = tuple(
        claim.min_ttc_s
        for claim in agent_claims
        if claim.min_ttc_s is not None and isfinite(claim.min_ttc_s)
    )
    finite_margins = tuple(
        claim.stopping_margin_m
        for claim in agent_claims
        if claim.stopping_margin_m is not None and isfinite(claim.stopping_margin_m)
    )
    probabilities = tuple(
        claim.probability for claim in agent_claims if claim.probability is not None
    )
    return _AggregatedClaim(
        agent_id=agent_id,
        min_ttc_s=min(finite_ttc, default=None),
        stopping_margin_m=min(finite_margins, default=None),
        probability=max(probabilities, default=None),
        confidence=min(claim.confidence for claim in agent_claims),
        severity=max(claim.severity for claim in agent_claims),
        recommended_max_speed_mps=min(
            claim.recommended_max_speed_mps for claim in agent_claims
        ),
        hard_stop_required=any(claim.hard_stop_required for claim in agent_claims),
    )


def _unit(value: float, maximum: float) -> float:
    """Normalize a non-negative value into the unit interval."""

    return min(max(value / maximum, 0.0), 1.0)


def _signed(value: float, maximum: float) -> float:
    """Normalize a signed value into the signed unit interval."""

    return min(max(value / maximum, -1.0), 1.0)


def _ttc(value: float | None, maximum: float) -> float:
    """Normalize TTC, treating an unavailable TTC as safely unbounded."""

    return 1.0 if value is None else _unit(value, maximum)


class ObservationBuilder:
    """Build the exact fixed-layout observation consumed by the Coordinator."""

    def __init__(self, config: ObservationConfig) -> None:
        self._config = config

    def build(
        self,
        snapshot: SceneObservation,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> NDArray[np.float32]:
        """Return the finite, bounded, fixed 24-slot observation."""

        self._validate(snapshot, claims, review)
        nominal = aggregate_agent_claims("nominal", claims)
        hazard = aggregate_agent_claims("hazard", claims)
        rule = aggregate_agent_claims("rule", claims)
        ego = snapshot.ego
        target = target_speed_mps(
            snapshot.previous_executed_action,
            ego.speed_mps,
            ego.speed_limit_mps,
        )
        supported = set(review.supported_agent_ids)

        values = np.asarray(
            [
                _unit(ego.speed_mps, self._config.max_speed_mps),
                self._speed_ratio(target, ego.speed_limit_mps),
                _signed(ego.acceleration_mps2, self._config.max_abs_acceleration_mps2),
                _signed(ego.lane_offset_m, self._config.max_abs_lane_offset_m),
                _unit(ego.route_progress, 1.0),
                _unit(ego.speed_limit_mps, self._config.max_speed_mps),
                0.0 if nominal is None else _ttc(nominal.min_ttc_s, self._config.max_ttc_s),
                1.0 if nominal is None else _unit(nominal.probability or 0.0, 1.0),
                0.0 if nominal is None else _unit(nominal.confidence, 1.0),
                0.0
                if nominal is None
                else _unit(nominal.recommended_max_speed_mps, self._config.max_speed_mps),
                0.0 if hazard is None else _ttc(hazard.min_ttc_s, self._config.max_ttc_s),
                -1.0
                if hazard is None or hazard.stopping_margin_m is None
                else _signed(hazard.stopping_margin_m, self._config.max_abs_stopping_margin_m),
                1.0 if hazard is None else _unit(hazard.severity, 1.0),
                0.0 if hazard is None else _unit(hazard.confidence, 1.0),
                0.0
                if hazard is None
                else _unit(hazard.recommended_max_speed_mps, self._config.max_speed_mps),
                0.0
                if rule is None
                else _unit(rule.recommended_max_speed_mps, self._config.max_speed_mps),
                1.0 if rule is None or rule.hard_stop_required else 0.0,
                float(
                    rule is None
                    or snapshot.road_context.stop_required
                    or snapshot.road_context.intersection_entry_prohibited
                ),
                _unit(review.conflict_score, 1.0),
                float(review.unresolved_conflict),
                _unit(len(supported.intersection(_REQUIRED_AGENT_IDS)) / 3.0, 1.0),
                _unit(review.max_severity, 1.0),
                _unit(float(snapshot.previous_executed_action), 3.0),
                float(snapshot.previous_shield_intervention),
            ],
            dtype=np.float32,
        )
        if values.shape != (24,) or not np.isfinite(values).all():
            raise ValueError("observation must contain 24 finite values")
        return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _speed_ratio(target_speed_mps: float, speed_limit_mps: float) -> float:
        if speed_limit_mps == 0.0:
            return 0.0
        return _unit(target_speed_mps / speed_limit_mps, 1.0)

    @staticmethod
    def _validate(
        snapshot: SceneObservation,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> None:
        if not valid_snapshot(snapshot):
            raise ValueError("invalid snapshot")
        if not valid_review(review):
            raise ValueError("invalid review")
        if any(not valid_claim(claim) for claim in claims):
            raise ValueError("invalid claim")
        supported_ids = review.supported_agent_ids
        if not all(isinstance(agent_id, str) and agent_id for agent_id in supported_ids):
            raise ValueError("invalid review")
        if len(supported_ids) != len(set(supported_ids)):
            raise ValueError("duplicate agent_id in review")
        speed_values = (snapshot.ego.speed_mps, snapshot.ego.speed_limit_mps)
        if not all(isfinite(value) for value in speed_values):
            raise ValueError("invalid snapshot")
