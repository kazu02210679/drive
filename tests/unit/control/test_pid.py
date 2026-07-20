import math

import pytest

from mad_driving.control import BoundedPID


def test_pid_uses_time_scaled_integral_and_derivative() -> None:
    pid = BoundedPID(kp=1.0, ki=1.0, kd=1.0, integral_limit=10.0)
    first = pid.update(error=2.0, dt_s=0.5, lower=-10.0, upper=10.0)
    second = pid.update(error=3.0, dt_s=0.5, lower=-10.0, upper=10.0)
    assert first == pytest.approx(3.0)
    assert second == pytest.approx(7.5)


def test_upper_saturation_blocks_windup_but_allows_unwinding() -> None:
    pid = BoundedPID(kp=1.0, ki=1.0, kd=0.0, integral_limit=10.0)
    assert pid.update(10.0, 1.0, -1.0, 1.0) == 1.0
    assert pid.integral == 0.0
    pid.update(-0.5, 1.0, -1.0, 1.0)
    assert pid.integral == -0.5


def test_lower_saturation_blocks_windup_but_allows_unwinding() -> None:
    pid = BoundedPID(kp=1.0, ki=1.0, kd=0.0, integral_limit=10.0)
    assert pid.update(-10.0, 1.0, -1.0, 1.0) == -1.0
    assert pid.integral == 0.0
    pid.update(0.5, 1.0, -1.0, 1.0)
    assert pid.integral == 0.5


def test_integral_is_clipped_to_configured_limit() -> None:
    pid = BoundedPID(kp=0.0, ki=1.0, kd=0.0, integral_limit=1.0)
    assert pid.update(2.0, 1.0, -10.0, 10.0) == 1.0
    assert pid.integral == 1.0


def test_reset_restores_first_update_behavior() -> None:
    pid = BoundedPID(1.0, 1.0, 1.0, 10.0)
    expected = pid.update(2.0, 0.5, -10.0, 10.0)
    pid.update(4.0, 0.5, -10.0, 10.0)
    pid.reset()
    assert pid.integral == 0.0
    assert pid.previous_error is None
    assert pid.update(2.0, 0.5, -10.0, 10.0) == expected


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_pid_rejects_non_finite_update_values(value: float) -> None:
    pid = BoundedPID(1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        pid.update(value, 0.1, -1.0, 1.0)


@pytest.mark.parametrize(
    ("dt_s", "lower", "upper"),
    [(0.0, -1.0, 1.0), (-0.1, -1.0, 1.0), (0.1, 2.0, 1.0)],
)
def test_pid_rejects_invalid_update_bounds(
    dt_s: float, lower: float, upper: float
) -> None:
    with pytest.raises(ValueError):
        BoundedPID(1.0, 0.0, 0.0, 1.0).update(1.0, dt_s, lower, upper)


@pytest.mark.parametrize(
    "arguments",
    [
        (-1.0, 0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0, 1.0),
        (0.0, 0.0, -1.0, 1.0),
        (0.0, 0.0, 0.0, 0.0),
        (math.nan, 0.0, 0.0, 1.0),
        (0.0, math.inf, 0.0, 1.0),
    ],
)
def test_pid_rejects_invalid_constructor_values(
    arguments: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValueError):
        BoundedPID(*arguments)
