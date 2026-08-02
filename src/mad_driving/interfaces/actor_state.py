"""State of a non-ego actor."""

from dataclasses import dataclass
from typing import Literal

from mad_driving.interfaces._validation import (
    require_finite,
    require_finite_values,
    require_non_empty,
    require_positive,
)

ActorType = Literal["vehicle", "crossing_actor", "obstacle"]


@dataclass(frozen=True)
class ActorState:
    actor_id: str
    actor_type: ActorType
    position_xy_m: tuple[float, float]
    velocity_xy_mps: tuple[float, float]
    acceleration_xy_mps2: tuple[float, float]
    heading_rad: float
    length_m: float
    width_m: float
    relative_longitudinal_m: float
    relative_lateral_m: float
    same_lane: bool
    visible: bool
    occluded: bool

    def __post_init__(self) -> None:
        require_non_empty("actor_id", self.actor_id)
        if self.actor_type not in ("vehicle", "crossing_actor", "obstacle"):
            raise ValueError("actor_type must be vehicle, crossing_actor, or obstacle")
        require_finite_values("position_xy_m", self.position_xy_m, length=2)
        require_finite_values("velocity_xy_mps", self.velocity_xy_mps, length=2)
        require_finite_values("acceleration_xy_mps2", self.acceleration_xy_mps2, length=2)
        require_finite("heading_rad", self.heading_rad)
        require_positive("length_m", self.length_m)
        require_positive("width_m", self.width_m)
        require_finite("relative_longitudinal_m", self.relative_longitudinal_m)
        require_finite("relative_lateral_m", self.relative_lateral_m)
