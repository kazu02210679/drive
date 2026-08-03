import math

import pytest

from mad_driving.control import (
    DrivingAction,
    action_for_speed_cap,
    target_speed_mps,
)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (DrivingAction.KEEP, 20.0),
        (DrivingAction.SLOW, 10.0),
        (DrivingAction.PREPARE_STOP, 5.0),
        (DrivingAction.STOP, 0.0),
    ],
)
def test_target_speed_mapping(action: DrivingAction, expected: float) -> None:
    assert target_speed_mps(action, 10.0, 20.0) == expected


@pytest.mark.parametrize(
    ("recommended", "expected"),
    [
        (0.0, DrivingAction.STOP),
        (5.0, DrivingAction.PREPARE_STOP),
        (5.0001, DrivingAction.SLOW),
        (12.0, DrivingAction.SLOW),
        (12.0001, DrivingAction.KEEP),
    ],
)
def test_speed_cap_boundaries(recommended: float, expected: DrivingAction) -> None:
    assert action_for_speed_cap(recommended, 20.0) is expected


def test_zero_speed_limit_requires_stop() -> None:
    assert action_for_speed_cap(10.0, 0.0) is DrivingAction.STOP


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_mapping_rejects_invalid_speeds(value: float) -> None:
    with pytest.raises(ValueError):
        target_speed_mps(DrivingAction.KEEP, value, 10.0)
