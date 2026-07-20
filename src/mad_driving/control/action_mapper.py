"""Pure mappings between high-level actions and physical speed targets."""

from math import isfinite

from mad_driving.control.actions import DrivingAction


def _speed(name: str, value: float) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def target_speed_mps(
    action: DrivingAction | int,
    current_speed_mps: float,
    speed_limit_mps: float,
) -> float:
    """Map an action to a target speed without requesting acceleration to slow."""

    action = DrivingAction(action)
    current = _speed("current_speed_mps", current_speed_mps)
    limit = _speed("speed_limit_mps", speed_limit_mps)
    if action is DrivingAction.KEEP:
        return limit
    if action is DrivingAction.SLOW:
        return min(current, 0.60 * limit)
    if action is DrivingAction.PREPARE_STOP:
        return min(current, 0.25 * limit)
    return 0.0


def action_for_speed_cap(
    recommended_max_speed_mps: float,
    speed_limit_mps: float,
) -> DrivingAction:
    """Convert a maximum safe speed to the least restrictive matching action."""

    recommended = _speed("recommended_max_speed_mps", recommended_max_speed_mps)
    limit = _speed("speed_limit_mps", speed_limit_mps)
    if limit == 0.0 or recommended <= 0.0:
        return DrivingAction.STOP
    if recommended <= 0.25 * limit:
        return DrivingAction.PREPARE_STOP
    if recommended <= 0.60 * limit:
        return DrivingAction.SLOW
    return DrivingAction.KEEP
