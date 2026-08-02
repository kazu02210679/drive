"""Small validation and conversion helpers for simulator state."""

from collections.abc import Sequence
from math import isfinite
from typing import Any, Protocol


class ConfigReader(Protocol):
    """Structural interface shared by dict and MetaDrive's Config class."""

    def get(self, key: str, default: Any = None) -> Any: ...


def finite_float(name: str, value: Any) -> float:
    """Convert a simulator scalar to a finite float."""

    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def xy_pair(name: str, value: Sequence[Any]) -> tuple[float, float]:
    """Convert a simulator vector to a finite two-dimensional tuple."""

    if len(value) < 2:
        raise ValueError(f"{name} must contain at least two values")
    return finite_float(f"{name}[0]", value[0]), finite_float(f"{name}[1]", value[1])


def decision_interval_s(config: ConfigReader) -> float:
    """Return one policy decision interval from MetaDrive timing settings."""

    physics_step = finite_float(
        "physics_world_step_size", config.get("physics_world_step_size", 0.02)
    )
    decision_repeat = finite_float("decision_repeat", config.get("decision_repeat", 5))
    interval = physics_step * decision_repeat
    if interval <= 0.0:
        raise ValueError("decision interval must be positive")
    return interval
