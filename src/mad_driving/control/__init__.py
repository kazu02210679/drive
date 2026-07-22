"""High-level actions and deterministic speed mappings."""

from typing import TYPE_CHECKING

from mad_driving.control.action_mapper import action_for_speed_cap, target_speed_mps
from mad_driving.control.actions import DrivingAction
from mad_driving.control.pid import BoundedPID

if TYPE_CHECKING:
    from mad_driving.control.lane_keeping_policy import LaneKeepingLongitudinalPolicy


def __getattr__(name: str) -> object:
    if name == "LaneKeepingLongitudinalPolicy":
        from mad_driving.control.lane_keeping_policy import LaneKeepingLongitudinalPolicy

        return LaneKeepingLongitudinalPolicy
    raise AttributeError(name)


__all__ = [
    "BoundedPID",
    "DrivingAction",
    "LaneKeepingLongitudinalPolicy",
    "action_for_speed_cap",
    "target_speed_mps",
]
