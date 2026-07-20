"""High-level longitudinal actions ordered by increasing conservatism."""

from enum import IntEnum


class DrivingAction(IntEnum):
    """Discrete speed decisions understood by the Phase 3 controller."""

    KEEP = 0
    SLOW = 1
    PREPARE_STOP = 2
    STOP = 3
