"""Deterministic construction and ordering helpers for specialist claims."""

import math
from collections.abc import Iterable
from dataclasses import replace

from mad_driving.interfaces import RiskClaim, SceneObservation


def claim_id(
    agent_id: str,
    observation: SceneObservation,
    event_type: str,
    target_actor_id: str | None,
) -> str:
    """Build a stable claim identifier without process-global state."""

    target = target_actor_id if target_actor_id is not None else "none"
    return f"{agent_id}:{observation.step_index}:{target}:{event_type}"


def claim_safety_key(claim: RiskClaim) -> tuple[object, ...]:
    """Order claims from most to least safety-critical with stable tie-breakers."""

    return (
        not claim.hard_stop_required,
        -claim.severity,
        claim.min_ttc_s is None,
        math.inf if claim.min_ttc_s is None else claim.min_ttc_s,
        math.inf if claim.stopping_margin_m is None else claim.stopping_margin_m,
        claim.recommended_max_speed_mps,
        claim.target_actor_id or "",
        claim.event_type,
    )


def ordered_claims(candidates: Iterable[RiskClaim], *, limit: int = 3) -> tuple[RiskClaim, ...]:
    """Retain a deterministic, uniquely identified safety-critical claim prefix."""

    if limit < 1:
        raise ValueError("limit must be positive")
    result: list[RiskClaim] = []
    claim_ids: set[str] = set()
    for claim in sorted(candidates, key=claim_safety_key):
        unique_claim = claim
        suffix = 2
        while unique_claim.claim_id in claim_ids:
            unique_claim = replace(claim, claim_id=f"{claim.claim_id}:{suffix}")
            suffix += 1
        result.append(unique_claim)
        claim_ids.add(unique_claim.claim_id)
        if len(result) == limit:
            break
    return tuple(result)


def neutral_claim(
    agent_id: str,
    observation: SceneObservation,
    event_type: str = "no_hazard",
) -> RiskClaim:
    """Return a finite no-hazard claim for an empty candidate set."""

    return RiskClaim(
        claim_id=claim_id(agent_id, observation, event_type, None),
        agent_id=agent_id,
        event_type=event_type,
        target_actor_id=None,
        probability=0.0,
        confidence=1.0,
        severity=0.0,
        time_horizon_s=0.0,
        min_ttc_s=None,
        stopping_margin_m=None,
        recommended_max_speed_mps=observation.ego.speed_limit_mps,
        hard_stop_required=False,
        evidence=("no_applicable_hazard",),
        assumptions=(),
        valid_until_step=observation.step_index,
    )
