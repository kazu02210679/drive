"""High-level actions and deterministic speed mappings."""

from mad_driving.control.action_mapper import action_for_speed_cap, target_speed_mps
from mad_driving.control.actions import DrivingAction

__all__ = ["DrivingAction", "action_for_speed_cap", "target_speed_mps"]
