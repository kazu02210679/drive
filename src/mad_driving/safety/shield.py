"""Deterministic final safety filter for high-level driving actions."""

from collections.abc import Sequence
from dataclasses import asdict

from mad_driving.config.models import ShieldConfig
from mad_driving.control import DrivingAction, action_for_speed_cap
from mad_driving.interfaces import (
    ActorState,
    EgoState,
    RiskClaim,
    SceneSnapshot,
    ShieldResult,
)

_REQUIRED_AGENT_IDS = frozenset({"nominal", "hazard", "rule"})


def _valid_claim(claim: RiskClaim) -> bool:
    try:
        RiskClaim(**asdict(claim))
    except (TypeError, ValueError):
        return False
    return True


def _valid_snapshot(snapshot: SceneSnapshot) -> bool:
    try:
        values = asdict(snapshot)
        ego = EgoState(**values.pop("ego"))
        actors = tuple(ActorState(**actor) for actor in values.pop("actors"))
        SceneSnapshot(ego=ego, actors=actors, **values)
    except (TypeError, ValueError):
        return False
    return True


class SafetyShield:
    """Apply fixed safety rules without relaxing the requested action."""

    def __init__(self, config: ShieldConfig) -> None:
        self._config = config

    def filter(
        self,
        requested_action: DrivingAction | int,
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
    ) -> ShieldResult:
        """Diagnose risk and optionally enforce the safest candidate action."""

        requested = DrivingAction(requested_action)
        if self._config.mode == "off":
            return ShieldResult(requested, requested, requested, False, False, ())

        valid_claims = tuple(claim for claim in claims if _valid_claim(claim))
        snapshot_is_valid = _valid_snapshot(snapshot)
        reasons: list[str] = []
        candidates = [DrivingAction.KEEP]

        if len(valid_claims) != len(claims) or not snapshot_is_valid:
            reasons.append("invalid_input")
            candidates.append(DrivingAction.STOP)
        if snapshot_is_valid and snapshot.collision_occurred:
            reasons.append("collision_occurred")
            candidates.append(DrivingAction.STOP)
        if snapshot_is_valid and snapshot.off_road:
            reasons.append("off_road")
            candidates.append(DrivingAction.STOP)
        if any(claim.hard_stop_required for claim in valid_claims):
            reasons.append("hard_stop_required")
            candidates.append(DrivingAction.STOP)

        present_agent_ids = {claim.agent_id for claim in valid_claims}
        missing_count = len(_REQUIRED_AGENT_IDS - present_agent_ids)
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

        if snapshot_is_valid:
            claim_action = max(
                (
                    action_for_speed_cap(
                        claim.recommended_max_speed_mps,
                        snapshot.ego.speed_limit_mps,
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
