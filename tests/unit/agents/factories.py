from typing import Any

from mad_driving.interfaces import ActorState, EgoState, RiskClaim, SceneSnapshot


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
) -> SceneSnapshot:
    values: dict[str, Any] = {
        "step_index": step_index,
        "sim_time_s": step_index * 0.1,
        "scenario_id": "phase2_unit",
        "seed": 42,
        "ego": make_ego(speed_mps=ego_speed_mps, speed_limit_mps=speed_limit_mps),
        "actors": actors,
        "stop_required": False,
        "occlusion_present": False,
        "distance_to_conflict_point_m": None,
        "previous_action": 0,
        "previous_shield_intervention": False,
        "collision_occurred": False,
        "off_road": False,
        "intersection_entry_prohibited": False,
    }
    values.update(overrides)
    return SceneSnapshot(**values)
