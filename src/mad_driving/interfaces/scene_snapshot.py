"""Immutable agent-visible ego and scene observations."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

from mad_driving.interfaces._validation import (
    require_action,
    require_finite,
    require_finite_values,
    require_non_negative,
    require_probability,
)
from mad_driving.interfaces.actor_state import ActorState
from mad_driving.interfaces.scene_frame import OcclusionRegion, RoadContext


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
        if not -pi <= self.heading_rad < pi:
            raise ValueError("heading_rad must be normalized to [-pi, pi)")
        require_finite("lane_offset_m", self.lane_offset_m)
        require_probability("route_progress", self.route_progress)
        require_non_negative("speed_limit_mps", self.speed_limit_mps)


@dataclass(frozen=True)
class SceneObservation:
    step_index: int
    sim_time_s: float
    ego: EgoState
    visible_actors: tuple[ActorState, ...]
    occlusion_regions: tuple[OcclusionRegion, ...]
    road_context: RoadContext
    previous_executed_action: int
    previous_shield_intervention: bool

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        require_non_negative("sim_time_s", self.sim_time_s)
        if not isinstance(self.ego, EgoState):
            raise ValueError("ego must be an EgoState instance")
        visible_actors = tuple(self.visible_actors)
        if not all(isinstance(actor, ActorState) and actor.visible for actor in visible_actors):
            raise ValueError("visible_actors must contain only visible ActorState values")
        occlusion_regions = tuple(self.occlusion_regions)
        if not all(isinstance(region, OcclusionRegion) for region in occlusion_regions):
            raise ValueError("occlusion_regions must contain only OcclusionRegion values")
        region_ids = tuple(region.region_id for region in occlusion_regions)
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("occlusion region_id values must be unique")
        if not isinstance(self.road_context, RoadContext):
            raise ValueError("road_context must be a RoadContext instance")
        require_action("previous_executed_action", self.previous_executed_action)
        object.__setattr__(self, "visible_actors", visible_actors)
        object.__setattr__(self, "occlusion_regions", occlusion_regions)


# Temporary import compatibility only. New production annotations use SceneObservation.
SceneSnapshot = SceneObservation
