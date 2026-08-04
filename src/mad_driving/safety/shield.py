"""Deterministic final safety filter for high-level driving actions."""

from collections.abc import Sequence

from mad_driving.config.models import ShieldConfig
from mad_driving.control import DrivingAction, action_for_speed_cap
from mad_driving.interfaces import RiskClaim, SceneObservation, ShieldResult
from mad_driving.interfaces.defensive_validation import (
    valid_claim,
    valid_snapshot,
)

_SPECIALIST_AGENT_IDS = frozenset({"nominal", "hazard", "rule"})


class SafetyShield:
    """Apply fixed safety rules without relaxing the requested action."""

    def __init__(self, config: ShieldConfig) -> None:
        self._config = config

    def filter(
        self,
        requested_action: DrivingAction | int,
        observation: SceneObservation,
        claims: Sequence[RiskClaim],
        *,
        expected_agent_ids: Sequence[str] = ("nominal", "hazard", "rule"),
        failed_agent_ids: Sequence[str] = (),
    ) -> ShieldResult:
        """Diagnose risk and optionally enforce the safest candidate action."""

        requested = DrivingAction(requested_action)
        if self._config.mode == "off":
            return ShieldResult(requested, requested, requested, False, False, ())

        expected = self._agent_id_set(expected_agent_ids, "expected_agent_ids")
        failed = self._agent_id_set(failed_agent_ids, "failed_agent_ids")
        if not failed.issubset(expected):
            raise ValueError("failed_agent_ids must be a subset of expected_agent_ids")
        valid_claims = tuple(claim for claim in claims if valid_claim(claim))
        observation_is_valid = valid_snapshot(observation)
        reasons: list[str] = []
        candidates = [DrivingAction.KEEP]

        unexpected_claim = any(claim.agent_id not in expected for claim in valid_claims)
        failed_claim = any(claim.agent_id in failed for claim in valid_claims)
        if (
            len(valid_claims) != len(claims)
            or not observation_is_valid
            or unexpected_claim
            or failed_claim
        ):
            reasons.append("invalid_input")
            candidates.append(DrivingAction.STOP)
        if any(claim.hard_stop_required for claim in valid_claims):
            reasons.append("hard_stop_required")
            candidates.append(DrivingAction.STOP)

        present_agent_ids = {claim.agent_id for claim in valid_claims if claim.agent_id in expected}
        missing_count = len((expected - present_agent_ids) | failed)
        if missing_count >= 2:
            reasons.append("multiple_agents_missing")
            candidates.append(DrivingAction(self._config.multiple_missing_action))
        elif missing_count == 1:
            reasons.append("agent_missing")
            candidates.append(DrivingAction(self._config.missing_agent_action))

        ttc_values = tuple(claim.min_ttc_s for claim in valid_claims if claim.min_ttc_s is not None)
        minimum_ttc = min(ttc_values, default=None)
        if minimum_ttc is not None and minimum_ttc <= self._config.imminent_ttc_s:
            reasons.append("imminent_ttc")
            candidates.append(DrivingAction.STOP)

        margin_values = tuple(
            claim.stopping_margin_m for claim in valid_claims if claim.stopping_margin_m is not None
        )
        minimum_margin = min(margin_values, default=None)
        if minimum_margin is not None and minimum_margin < self._config.emergency_margin_m:
            reasons.append("negative_stopping_margin")
            candidates.append(DrivingAction.STOP)

        if (
            minimum_ttc is not None
            and minimum_ttc > self._config.imminent_ttc_s
            and minimum_ttc <= self._config.caution_ttc_s
        ):
            reasons.append("caution_ttc")
            candidates.append(DrivingAction.PREPARE_STOP)

        if (
            minimum_margin is not None
            and minimum_margin >= self._config.emergency_margin_m
            and minimum_margin < self._config.caution_margin_m
        ):
            reasons.append("low_stopping_margin")
            candidates.append(DrivingAction.PREPARE_STOP)

        if observation_is_valid:
            claim_action = max(
                (
                    action_for_speed_cap(
                        claim.recommended_max_speed_mps,
                        observation.ego.speed_limit_mps,
                    )
                    for claim in valid_claims
                ),
                default=DrivingAction.KEEP,
            )
            if claim_action > requested:
                reasons.append("claim_speed_limit")
                candidates.append(claim_action)

        required = max(candidates)
        executed = requested if self._config.mode == "monitor" else max(requested, required)
        return ShieldResult(
            requested_action=requested,
            required_action=required,
            executed_action=executed,
            intervention_required=required > requested,
            intervened=executed != requested,
            reasons=tuple(reasons),
        )

    @staticmethod
    def _agent_id_set(agent_ids: Sequence[str], field_name: str) -> frozenset[str]:
        if isinstance(agent_ids, str | bytes):
            raise ValueError(f"{field_name} must be a sequence of specialist agent IDs")
        values = tuple(agent_ids)
        if not all(isinstance(agent_id, str) and agent_id for agent_id in values):
            raise ValueError(f"{field_name} must contain non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError(f"{field_name} must contain unique values")
        result = frozenset(values)
        if not result.issubset(_SPECIALIST_AGENT_IDS):
            raise ValueError(f"{field_name} contains an unknown specialist agent ID")
        return result
