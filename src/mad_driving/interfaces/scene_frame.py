"""Immutable scene context and privileged simulator state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from mad_driving.interfaces._validation import (
    require_finite,
    require_finite_values,
    require_non_empty,
)
from mad_driving.interfaces.actor_state import ActorState

if TYPE_CHECKING:
    from mad_driving.interfaces.scene_snapshot import SceneObservation
    from mad_driving.scenarios.seeding import EpisodeSeeds


CollisionKind = Literal[
    "vehicle",
    "crossing_actor",
    "object",
    "sidewalk",
    "building",
]


@dataclass(frozen=True)
class OcclusionRegion:
    """A named boundary that can hide actors from the ego vehicle."""

    region_id: str
    boundary_points_xy_m: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        require_non_empty("region_id", self.region_id)
        boundary_points = tuple(tuple(point) for point in self.boundary_points_xy_m)
        if len(boundary_points) < 2:
            raise ValueError("boundary_points_xy_m must contain at least two points")
        for point in boundary_points:
            require_finite_values("boundary_points_xy_m", point, length=2)
        object.__setattr__(self, "boundary_points_xy_m", boundary_points)


@dataclass(frozen=True)
class RoadContext:
    """Scenario-defined road rules relevant to the current observation."""

    stop_required: bool
    distance_to_conflict_point_m: float | None
    intersection_entry_prohibited: bool

    def __post_init__(self) -> None:
        if self.distance_to_conflict_point_m is not None:
            require_finite("distance_to_conflict_point_m", self.distance_to_conflict_point_m)


@dataclass(frozen=True)
class PrivilegedWorldState:
    """Simulator truth reserved for reward, evaluation, and debug outputs."""

    all_actors: tuple[ActorState, ...]
    collision_occurred: bool
    collision_kind: CollisionKind | None
    off_road: bool
    arrived: bool
    scenario_success: bool
    scenario_failure: bool

    def __post_init__(self) -> None:
        all_actors = tuple(self.all_actors)
        if not all(isinstance(actor, ActorState) for actor in all_actors):
            raise ValueError("all_actors must contain only ActorState values")
        if self.collision_kind not in {
            None,
            "vehicle",
            "crossing_actor",
            "object",
            "sidewalk",
            "building",
        }:
            raise ValueError("collision_kind is not recognized")
        object.__setattr__(self, "all_actors", all_actors)


@dataclass(frozen=True)
class SceneFrame:
    """One decision-boundary view split between observation and simulator truth."""

    scenario_id: str
    seeds: EpisodeSeeds
    observation: SceneObservation
    privileged: PrivilegedWorldState

    def __post_init__(self) -> None:
        from mad_driving.interfaces.scene_snapshot import SceneObservation
        from mad_driving.scenarios.seeding import EpisodeSeeds

        require_non_empty("scenario_id", self.scenario_id)
        if not isinstance(self.seeds, EpisodeSeeds):
            raise ValueError("seeds must be an EpisodeSeeds instance")
        if not isinstance(self.observation, SceneObservation):
            raise ValueError("observation must be a SceneObservation")
        if not isinstance(self.privileged, PrivilegedWorldState):
            raise ValueError("privileged must be a PrivilegedWorldState")
