from typing import Any

from mad_driving.agents.suite import AgentAnalysisResult
from mad_driving.interfaces import (
    ActorState,
    CriticReview,
    EgoState,
    OcclusionRegion,
    PrivilegedWorldState,
    RiskClaim,
    RoadContext,
    SceneFrame,
    SceneObservation,
)
from mad_driving.scenarios import EpisodeSeeds


def make_claim(agent_id: str = "nominal", **overrides: Any) -> RiskClaim:
    values: dict[str, Any] = {
        "claim_id": f"{agent_id}:1:none:test",
        "agent_id": agent_id,
        "event_type": "test",
        "target_actor_id": None,
        "probability": 0.0,
        "confidence": 1.0,
        "severity": 0.0,
        "time_horizon_s": 1.0,
        "min_ttc_s": None,
        "stopping_margin_m": None,
        "recommended_max_speed_mps": 20.0,
        "hard_stop_required": False,
        "evidence": ("test",),
        "assumptions": (),
        "valid_until_step": 1,
    }
    values.update(overrides)
    return RiskClaim(**values)


def make_analysis(
    *,
    claims: tuple[RiskClaim, ...] = (),
    failed_agent_ids: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
    review: CriticReview | None = None,
    expected_agent_ids: tuple[str, ...] = ("nominal", "hazard", "rule"),
) -> AgentAnalysisResult:
    """Build a validated neutral analysis result for non-agent test seams."""

    if review is None:
        review = CriticReview(
            conflict_score=0.0,
            unresolved_conflict=False,
            max_severity=0.0,
            supported_agent_ids=tuple(sorted({claim.agent_id for claim in claims})),
            challenged_claim_ids=(),
            reasons=(),
        )
    return AgentAnalysisResult(
        claims=claims,
        failed_agent_ids=failed_agent_ids,
        errors=errors,
        review=review,
        expected_agent_ids=expected_agent_ids,
    )


def make_actor(
    actor_id: str = "actor-1",
    *,
    actor_type: str = "vehicle",
    longitudinal_m: float = 20.0,
    lateral_m: float = 0.0,
    longitudinal_speed_mps: float = 5.0,
    lateral_speed_mps: float = 0.0,
    longitudinal_acceleration_mps2: float = 0.0,
    lateral_acceleration_mps2: float = 0.0,
    same_lane: bool = True,
    length_m: float = 4.5,
    width_m: float = 1.8,
) -> ActorState:
    return ActorState(
        actor_id=actor_id,
        actor_type=actor_type,  # type: ignore[arg-type]
        position_xy_m=(longitudinal_m, lateral_m),
        velocity_xy_mps=(longitudinal_speed_mps, lateral_speed_mps),
        acceleration_xy_mps2=(
            longitudinal_acceleration_mps2,
            lateral_acceleration_mps2,
        ),
        heading_rad=0.0,
        length_m=length_m,
        width_m=width_m,
        relative_longitudinal_m=longitudinal_m,
        relative_lateral_m=lateral_m,
        same_lane=same_lane,
        visible=True,
        occluded=False,
    )


def make_occlusion(region_id: str = "unit-occlusion") -> OcclusionRegion:
    return OcclusionRegion(
        region_id=region_id,
        boundary_points_xy_m=((0.0, 0.0), (1.0, 0.0)),
    )


def make_ego(*, speed_mps: float = 10.0, speed_limit_mps: float = 15.0) -> EgoState:
    return EgoState(
        position_xy_m=(0.0, 0.0),
        speed_mps=speed_mps,
        acceleration_mps2=0.0,
        heading_rad=0.0,
        lane_offset_m=0.0,
        route_progress=0.25,
        speed_limit_mps=speed_limit_mps,
    )


def make_snapshot(
    *,
    step_index: int = 1,
    ego_speed_mps: float = 10.0,
    speed_limit_mps: float = 15.0,
    actors: tuple[ActorState, ...] = (),
    **overrides: Any,
) -> SceneObservation:
    values: dict[str, Any] = {
        "step_index": step_index,
        "sim_time_s": step_index * 0.1,
        "ego": make_ego(speed_mps=ego_speed_mps, speed_limit_mps=speed_limit_mps),
        "visible_actors": actors,
        "occlusion_regions": (),
        "road_context": RoadContext(
            stop_required=False,
            distance_to_conflict_point_m=None,
            intersection_entry_prohibited=False,
        ),
        "previous_executed_action": 0,
        "previous_shield_intervention": False,
    }
    values.update(overrides)
    return SceneObservation(**values)


def make_frame(
    *,
    scenario_id: str = "phase2_unit",
    seeds: EpisodeSeeds | None = None,
    observation: SceneObservation | None = None,
    all_actors: tuple[ActorState, ...] = (),
    collision_occurred: bool = False,
    collision_kind: str | None = None,
    off_road: bool = False,
    arrived: bool = False,
    scenario_success: bool = False,
    scenario_failure: bool = False,
    minimum_actual_ttc_s: float | None = None,
    hard_rule_constraint: bool = False,
) -> SceneFrame:
    """Build a full frame for reward and environment tests."""

    return SceneFrame(
        scenario_id=scenario_id,
        seeds=seeds
        or EpisodeSeeds(
            episode_rng_seed=42,
            metadrive_scenario_index=7,
            scenario_parameter_seed=11,
        ),
        observation=make_snapshot() if observation is None else observation,
        privileged=PrivilegedWorldState(
            all_actors=all_actors,
            collision_occurred=collision_occurred,
            collision_kind=collision_kind,  # type: ignore[arg-type]
            off_road=off_road,
            arrived=arrived,
            scenario_success=scenario_success,
            scenario_failure=scenario_failure,
            minimum_actual_ttc_s=minimum_actual_ttc_s,
            hard_rule_constraint=hard_rule_constraint,
        ),
    )
