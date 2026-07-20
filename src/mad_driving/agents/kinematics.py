"""Finite, side-effect-free kinematic primitives for driving agents."""

from math import cos, floor, hypot, isclose, sin, sqrt

from mad_driving.world_model.validation import finite_float, xy_pair


def _non_negative(name: str, value: float) -> float:
    result = finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _negative_deceleration(value: float) -> float:
    result = finite_float("deceleration_mps2", value)
    if result >= 0.0:
        raise ValueError("deceleration_mps2 must be strictly negative")
    return result


def sample_times(horizon_s: float, step_s: float) -> tuple[float, ...]:
    """Return regular positive samples plus the exact horizon."""

    horizon = finite_float("horizon_s", horizon_s)
    step = finite_float("step_s", step_s)
    if horizon <= 0.0:
        raise ValueError("horizon_s must be positive")
    if step <= 0.0:
        raise ValueError("step_s must be positive")

    count = floor(horizon / step)
    values = [index * step for index in range(1, count + 1)]
    if values and isclose(values[-1], horizon, rel_tol=0.0, abs_tol=1e-12):
        values[-1] = horizon
    else:
        values.append(horizon)
    return tuple(values)


def project_vector(vector_xy: tuple[float, float], heading_rad: float) -> tuple[float, float]:
    """Project a world-frame vector into longitudinal/lateral ego axes."""

    x, y = xy_pair("vector_xy", vector_xy)
    heading = finite_float("heading_rad", heading_rad)
    return (
        cos(heading) * x + sin(heading) * y,
        -sin(heading) * x + cos(heading) * y,
    )


def relative_position(
    initial_xy: tuple[float, float],
    relative_velocity_xy: tuple[float, float],
    relative_acceleration_xy: tuple[float, float],
    time_s: float,
) -> tuple[float, float]:
    """Project a relative constant-acceleration trajectory to one time."""

    initial = xy_pair("initial_xy", initial_xy)
    velocity = xy_pair("relative_velocity_xy", relative_velocity_xy)
    acceleration = xy_pair("relative_acceleration_xy", relative_acceleration_xy)
    time = _non_negative("time_s", time_s)
    return (
        initial[0] + velocity[0] * time + 0.5 * acceleration[0] * time**2,
        initial[1] + velocity[1] * time + 0.5 * acceleration[1] * time**2,
    )


def rectangular_clearance(
    relative_xy: tuple[float, float],
    longitudinal_envelope_m: float,
    lateral_envelope_m: float,
) -> float:
    """Return Euclidean distance outside a rectangular collision envelope."""

    longitudinal, lateral = xy_pair("relative_xy", relative_xy)
    longitudinal_envelope = _non_negative("longitudinal_envelope_m", longitudinal_envelope_m)
    lateral_envelope = _non_negative("lateral_envelope_m", lateral_envelope_m)
    outside_longitudinal = max(abs(longitudinal) - longitudinal_envelope, 0.0)
    outside_lateral = max(abs(lateral) - lateral_envelope, 0.0)
    return hypot(outside_longitudinal, outside_lateral)


def stopping_distance(speed_mps: float, deceleration_mps2: float) -> float:
    """Return braking distance for a strictly negative constant deceleration."""

    speed = _non_negative("speed_mps", speed_mps)
    deceleration = _negative_deceleration(deceleration_mps2)
    return speed**2 / (2.0 * abs(deceleration))


def safe_speed_for_distance(
    distance_m: float,
    reaction_s: float,
    deceleration_mps2: float,
) -> float:
    """Invert reaction-plus-braking distance for the non-negative speed root."""

    distance = finite_float("distance_m", distance_m)
    reaction = _non_negative("reaction_s", reaction_s)
    deceleration = _negative_deceleration(deceleration_mps2)
    if distance <= 0.0:
        return 0.0
    braking = abs(deceleration)
    return max(
        0.0,
        sqrt((braking * reaction) ** 2 + 2.0 * braking * distance) - braking * reaction,
    )
