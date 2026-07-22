"""Validated simulator-neutral contracts for scripted scenario actors."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Literal

from mad_driving.interfaces._validation import (
    require_finite,
    require_finite_values,
    require_non_empty,
    require_non_negative,
    require_positive,
)

LaneIndex = tuple[str, str, int]
ActorKind = Literal["vehicle", "crossing_actor", "occluder"]


def _validate_lane_index(lane_index: LaneIndex) -> None:
    if (
        not isinstance(lane_index, tuple)
        or len(lane_index) != 3
        or not isinstance(lane_index[0], str)
        or not isinstance(lane_index[1], str)
        or isinstance(lane_index[2], bool)
        or not isinstance(lane_index[2], int)
    ):
        raise ValueError("lane_index must be a (str, str, int) tuple")


@dataclass(frozen=True)
class RoadGeometry:
    """Ego-lane geometry sampled after the simulator has reset."""

    ego_lane_index: LaneIndex
    ego_longitudinal_m: float
    ego_lateral_m: float
    ego_speed_mps: float
    decision_interval_s: float

    def __post_init__(self) -> None:
        _validate_lane_index(self.ego_lane_index)
        require_finite("ego_longitudinal_m", self.ego_longitudinal_m)
        require_finite("ego_lateral_m", self.ego_lateral_m)
        require_non_negative("ego_speed_mps", self.ego_speed_mps)
        require_positive("decision_interval_s", self.decision_interval_s)


@dataclass(frozen=True)
class LaneVehicleSpawn:
    """One vehicle positioned in a procedural road lane."""

    actor_id: str
    lane_index: LaneIndex
    longitudinal_m: float
    lateral_m: float
    speed_mps: float

    def __post_init__(self) -> None:
        require_non_empty("actor_id", self.actor_id)
        _validate_lane_index(self.lane_index)
        require_finite("longitudinal_m", self.longitudinal_m)
        require_finite("lateral_m", self.lateral_m)
        require_non_negative("speed_mps", self.speed_mps)


@dataclass(frozen=True)
class KinematicActorSpawn:
    """One world-relative scripted actor with a finite initial velocity."""

    actor_id: str
    actor_kind: ActorKind
    position_xy_m: tuple[float, float]
    heading_rad: float
    velocity_xy_mps: tuple[float, float]

    def __post_init__(self) -> None:
        require_non_empty("actor_id", self.actor_id)
        if self.actor_kind not in ("vehicle", "crossing_actor", "occluder"):
            raise ValueError("actor_kind must be vehicle, crossing_actor, or occluder")
        require_finite_values("position_xy_m", self.position_xy_m, length=2)
        require_finite("heading_rad", self.heading_rad)
        if not -pi <= self.heading_rad < pi:
            raise ValueError("heading_rad must be normalized to [-pi, pi)")
        require_finite_values("velocity_xy_mps", self.velocity_xy_mps, length=2)


@dataclass(frozen=True)
class StaticOccluderSpawn:
    """One static world-relative occluder outside the ego collision corridor."""

    actor_id: str
    position_xy_m: tuple[float, float]
    heading_rad: float
    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        require_non_empty("actor_id", self.actor_id)
        require_finite_values("position_xy_m", self.position_xy_m, length=2)
        require_finite("heading_rad", self.heading_rad)
        if not -pi <= self.heading_rad < pi:
            raise ValueError("heading_rad must be normalized to [-pi, pi)")
        require_positive("length_m", self.length_m)
        require_positive("width_m", self.width_m)


@dataclass(frozen=True)
class ActorCommand:
    """A pending longitudinal acceleration command for one scripted actor."""

    longitudinal_acceleration_mps2: float

    def __post_init__(self) -> None:
        require_finite("longitudinal_acceleration_mps2", self.longitudinal_acceleration_mps2)

    @classmethod
    def longitudinal(cls, acceleration_mps2: float) -> ActorCommand:
        """Create one finite longitudinal acceleration command."""

        return cls(acceleration_mps2)


@dataclass(frozen=True)
class ScenarioActorState:
    """Finite dynamic state of a scenario-owned actor."""

    actor_id: str
    position_xy_m: tuple[float, float]
    velocity_xy_mps: tuple[float, float]
    acceleration_xy_mps2: tuple[float, float]
    heading_rad: float

    def __post_init__(self) -> None:
        require_non_empty("actor_id", self.actor_id)
        require_finite_values("position_xy_m", self.position_xy_m, length=2)
        require_finite_values("velocity_xy_mps", self.velocity_xy_mps, length=2)
        require_finite_values("acceleration_xy_mps2", self.acceleration_xy_mps2, length=2)
        require_finite("heading_rad", self.heading_rad)
        if not -pi <= self.heading_rad < pi:
            raise ValueError("heading_rad must be normalized to [-pi, pi)")
