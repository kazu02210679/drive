"""Immutable scene context and privileged simulator state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from mad_driving.interfaces._validation import (
    require_finite,
    require_finite_values,
    require_non_empty,
    require_non_negative,
    require_positive,
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


def stopping_margin_m(
    *,
    ego_speed_mps: float,
    minimum_actual_ttc_s: float | None,
    reaction_delay_s: float,
    safe_deceleration_mps2: float,
) -> float | None:
    """Return the fixed-oracle braking margin, or ``None`` without a TTC."""

    require_non_negative("ego_speed_mps", ego_speed_mps)
    require_non_negative("reaction_delay_s", reaction_delay_s)
    require_positive("safe_deceleration_mps2", safe_deceleration_mps2)
    if minimum_actual_ttc_s is None:
        return None
    require_non_negative("minimum_actual_ttc_s", minimum_actual_ttc_s)
    available_distance_m = ego_speed_mps * minimum_actual_ttc_s
    required_distance_m = (
        ego_speed_mps * reaction_delay_s
        + ego_speed_mps**2 / (2.0 * safe_deceleration_mps2)
    )
    return available_distance_m - required_distance_m


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
    minimum_actual_ttc_s: float | None
    minimum_actual_stopping_margin_m: float | None
    hard_rule_constraint: bool

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
        if self.minimum_actual_ttc_s is not None:
            require_finite("minimum_actual_ttc_s", self.minimum_actual_ttc_s)
            if self.minimum_actual_ttc_s < 0.0:
                raise ValueError("minimum_actual_ttc_s must be non-negative")
        if self.minimum_actual_stopping_margin_m is not None:
            require_finite(
                "minimum_actual_stopping_margin_m", self.minimum_actual_stopping_margin_m
            )
        if not isinstance(self.hard_rule_constraint, bool):
            raise ValueError("hard_rule_constraint must be boolean")
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
