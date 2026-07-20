"""Shared runtime validation for immutable interface dataclasses."""

import math
from collections.abc import Iterable


def require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def require_finite_values(name: str, values: Iterable[float], *, length: int | None = None) -> None:
    items = tuple(values)
    if length is not None and len(items) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    for value in items:
        require_finite(name, value)


def require_non_negative(name: str, value: float) -> None:
    require_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def require_positive(name: str, value: float) -> None:
    require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def require_probability(name: str, value: float) -> None:
    require_finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def require_action(name: str, value: int) -> None:
    if value not in range(4):
        raise ValueError(f"{name} must be an integer in [0, 3]")


def require_non_empty(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty")


def canonical_string_tuple(name: str, values: object) -> tuple[str, ...]:
    """Return an isolated immutable string sequence or reject malformed input."""

    if isinstance(values, str | bytes):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        items: tuple[object, ...] = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of strings") from error
    if not all(isinstance(item, str) for item in items):
        raise ValueError(f"{name} must contain only strings")
    return tuple(item for item in items if isinstance(item, str))
