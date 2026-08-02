"""Deterministic construction helpers shared by claim-producing agents."""

from mad_driving.interfaces import RiskClaim, SceneSnapshot


def claim_id(
    agent_id: str,
    snapshot: SceneSnapshot,
    event_type: str,
    target_actor_id: str | None,
) -> str:
    """Build a stable claim identifier without process-global state."""

    target = target_actor_id if target_actor_id is not None else "none"
    return f"{agent_id}:{snapshot.step_index}:{target}:{event_type}"


def neutral_claim(
    agent_id: str,
    snapshot: SceneSnapshot,
    event_type: str = "no_hazard",
) -> RiskClaim:
    """Return a finite no-hazard claim for an empty candidate set."""

    return RiskClaim(
        claim_id=claim_id(agent_id, snapshot, event_type, None),
        agent_id=agent_id,
        event_type=event_type,
        target_actor_id=None,
        probability=0.0,
        confidence=1.0,
        severity=0.0,
        time_horizon_s=0.0,
        min_ttc_s=None,
        stopping_margin_m=None,
        recommended_max_speed_mps=snapshot.ego.speed_limit_mps,
        hard_stop_required=False,
        evidence=("no_applicable_hazard",),
        assumptions=(),
        valid_until_step=snapshot.step_index,
    )
