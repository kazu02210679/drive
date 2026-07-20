"""Fixed 24-dimensional coordinator observation assembly."""

from collections.abc import Sequence
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from mad_driving.config.models import ObservationConfig
from mad_driving.control import target_speed_mps
from mad_driving.interfaces import CriticReview, RiskClaim, SceneSnapshot
from mad_driving.interfaces.defensive_validation import valid_claim, valid_review, valid_snapshot

_REQUIRED_AGENT_IDS = ("nominal", "hazard", "rule")


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
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> NDArray[np.float32]:
        """Return the finite, bounded, fixed 24-slot observation."""

        self._validate(snapshot, claims, review)
        indexed = self._claim_index(claims)
        nominal = indexed.get("nominal")
        hazard = indexed.get("hazard")
        rule = indexed.get("rule")
        ego = snapshot.ego
        target = target_speed_mps(snapshot.previous_action, ego.speed_mps, ego.speed_limit_mps)
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
                    or snapshot.stop_required
                    or snapshot.intersection_entry_prohibited
                    or snapshot.collision_occurred
                    or snapshot.off_road
                ),
                _unit(review.conflict_score, 1.0),
                float(review.unresolved_conflict),
                _unit(len(supported.intersection(_REQUIRED_AGENT_IDS)) / 3.0, 1.0),
                _unit(review.max_severity, 1.0),
                _unit(float(snapshot.previous_action), 3.0),
                float(snapshot.previous_shield_intervention),
            ],
            dtype=np.float32,
        )
        if values.shape != (24,) or not np.isfinite(values).all():
            raise ValueError("observation must contain 24 finite values")
        return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)

    def _claim_index(self, claims: Sequence[RiskClaim]) -> dict[str, RiskClaim]:
        indexed: dict[str, RiskClaim] = {}
        for claim in claims:
            if claim.agent_id in indexed:
                raise ValueError(f"duplicate agent_id: {claim.agent_id}")
            indexed[claim.agent_id] = claim
        return indexed

    @staticmethod
    def _speed_ratio(target_speed_mps: float, speed_limit_mps: float) -> float:
        if speed_limit_mps == 0.0:
            return 0.0
        return _unit(target_speed_mps / speed_limit_mps, 1.0)

    @staticmethod
    def _validate(
        snapshot: SceneSnapshot,
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
