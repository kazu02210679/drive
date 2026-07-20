import math

import pytest

from mad_driving.agents.kinematics import (
    project_vector,
    rectangular_clearance,
    relative_position,
    safe_speed_for_distance,
    sample_times,
    stopping_distance,
)


def test_sample_times_includes_regular_steps_and_exact_horizon() -> None:
    assert sample_times(1.0, 0.25) == (0.25, 0.5, 0.75, 1.0)
    assert sample_times(1.1, 0.25) == (0.25, 0.5, 0.75, 1.0, 1.1)


def test_project_vector_uses_ego_heading_frame() -> None:
    longitudinal, lateral = project_vector((1.0, 0.0), math.pi / 2.0)

    assert longitudinal == pytest.approx(0.0, abs=1e-12)
    assert lateral == pytest.approx(-1.0)


def test_relative_position_applies_velocity_and_acceleration() -> None:
    assert relative_position((10.0, 2.0), (-2.0, 1.0), (1.0, -2.0), 2.0) == pytest.approx(
        (8.0, 0.0)
    )


def test_rectangular_clearance_is_zero_inside_envelope() -> None:
    assert rectangular_clearance((2.0, 0.5), 2.5, 1.0) == 0.0


def test_rectangular_clearance_measures_distance_outside_both_axes() -> None:
    assert rectangular_clearance((4.0, 3.0), 2.0, 1.0) == pytest.approx(math.sqrt(8.0))


def test_stopping_distance_and_inverse_safe_speed_agree() -> None:
    reaction_s = 0.5
    distance = 12.0 * reaction_s + stopping_distance(12.0, -6.0)

    assert stopping_distance(0.0, -6.0) == 0.0
    assert safe_speed_for_distance(distance, reaction_s, -6.0) == pytest.approx(12.0)
    assert safe_speed_for_distance(-1.0, reaction_s, -6.0) == 0.0


@pytest.mark.parametrize("bad", [0.0, 1.0, math.nan, math.inf])
def test_stopping_distance_rejects_invalid_deceleration(bad: float) -> None:
    with pytest.raises(ValueError, match="deceleration"):
        stopping_distance(5.0, bad)


@pytest.mark.parametrize(
    ("call", "field"),
    [
        (lambda: sample_times(math.nan, 0.25), "horizon"),
        (lambda: sample_times(1.0, 0.0), "step"),
        (lambda: project_vector((math.inf, 0.0), 0.0), "vector"),
        (lambda: relative_position((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), math.nan), "time"),
        (lambda: rectangular_clearance((0.0, 0.0), -1.0, 1.0), "envelope"),
        (lambda: safe_speed_for_distance(1.0, -0.1, -6.0), "reaction"),
    ],
)
def test_kinematics_reject_invalid_inputs(call, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        call()
