"""Immutable ego and scene snapshots."""

from dataclasses import dataclass

from mad_driving.interfaces._validation import (
    require_action,
    require_finite,
    require_finite_values,
    require_non_empty,
    require_non_negative,
    require_probability,
)
from mad_driving.interfaces.actor_state import ActorState


@dataclass(frozen=True)
class EgoState:
    position_xy_m: tuple[float, float]
    speed_mps: float
    acceleration_mps2: float
    heading_rad: float
    lane_offset_m: float
    route_progress: float
    speed_limit_mps: float

    def __post_init__(self) -> None:
        require_finite_values("position_xy_m", self.position_xy_m, length=2)
        require_non_negative("speed_mps", self.speed_mps)
        require_finite("acceleration_mps2", self.acceleration_mps2)
        require_finite("heading_rad", self.heading_rad)
        require_finite("lane_offset_m", self.lane_offset_m)
        require_probability("route_progress", self.route_progress)
        require_non_negative("speed_limit_mps", self.speed_limit_mps)


@dataclass(frozen=True)
class SceneSnapshot:
    step_index: int
    sim_time_s: float
    scenario_id: str
    seed: int
    ego: EgoState
    actors: tuple[ActorState, ...]
    stop_required: bool
    occlusion_present: bool
    distance_to_conflict_point_m: float | None
    previous_action: int
    previous_shield_intervention: bool
    collision_occurred: bool
    off_road: bool
    intersection_entry_prohibited: bool

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        require_non_negative("sim_time_s", self.sim_time_s)
        require_non_empty("scenario_id", self.scenario_id)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.distance_to_conflict_point_m is not None:
            require_finite("distance_to_conflict_point_m", self.distance_to_conflict_point_m)
        require_action("previous_action", self.previous_action)
