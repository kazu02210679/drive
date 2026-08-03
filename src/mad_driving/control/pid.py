"""Small deterministic PID controller with bounded conditional integration."""

from math import isfinite


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _require_finite(*values: float) -> None:
    if not all(isfinite(value) for value in values):
        raise ValueError("PID values must be finite")


class BoundedPID:
    """Stateful PID with integral clipping and saturation anti-windup."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float,
    ) -> None:
        _require_finite(kp, ki, kd, integral_limit)
        if kp < 0.0 or ki < 0.0 or kd < 0.0 or integral_limit <= 0.0:
            raise ValueError("PID gains must be non-negative and limit must be positive")
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral_limit = integral_limit
        self._integral = 0.0
        self._previous_error: float | None = None

    @property
    def integral(self) -> float:
        """Return the accumulated, bounded error integral."""

        return self._integral

    @property
    def previous_error(self) -> float | None:
        """Return the error from the previous update, if any."""

        return self._previous_error

    def update(self, error: float, dt_s: float, lower: float, upper: float) -> float:
        """Advance one update and return an output clipped to the given bounds."""

        _require_finite(error, dt_s, lower, upper)
        if dt_s <= 0.0 or lower > upper:
            raise ValueError("invalid PID update bounds")
        derivative = 0.0 if self._previous_error is None else (error - self._previous_error) / dt_s
        candidate = _clip(
            self._integral + error * dt_s,
            -self._integral_limit,
            self._integral_limit,
        )
        raw = self._kp * error + self._ki * candidate + self._kd * derivative
        saturated_high = raw > upper
        saturated_low = raw < lower
        if not ((saturated_high and error > 0.0) or (saturated_low and error < 0.0)):
            self._integral = candidate
            raw = self._kp * error + self._ki * self._integral + self._kd * derivative
        self._previous_error = error
        return _clip(raw, lower, upper)

    def reset(self) -> None:
        """Clear all accumulated controller state."""

        self._integral = 0.0
        self._previous_error = None
