import math
from types import SimpleNamespace
from typing import Any

import pytest

from mad_driving.config.models import ControlConfig
from mad_driving.control import DrivingAction, LaneKeepingLongitudinalPolicy


class FakeLane:
    def __init__(self, *, heading: float = 0.0, speed_limit: float | None = 72.0):
        self.heading = heading
        if speed_limit is not None:
            self.speed_limit = speed_limit

    def local_coordinates(self, position: tuple[float, float]) -> tuple[float, float]:
        return position

    def heading_theta_at(self, longitudinal: float) -> float:
        del longitudinal
        return self.heading


class FakeVehicle:
    def __init__(self, lane: FakeLane | None = None) -> None:
        self.position = (5.0, 0.0)
        self.heading_theta = 0.0
        self.speed = 10.0
        self.max_speed_m_s = 15.0
        self.lane = lane if lane is not None else FakeLane()
        self.navigation = SimpleNamespace(current_lane=self.lane)


@pytest.fixture
def fake_policy() -> LaneKeepingLongitudinalPolicy:
    policy = object.__new__(LaneKeepingLongitudinalPolicy)
    policy._control_config = ControlConfig()
    policy.action_info = {}
    policy._build_controllers()
    return policy


@pytest.fixture
def fake_vehicle() -> FakeVehicle:
    return FakeVehicle()


def test_policy_input_space_is_discrete_four() -> None:
    assert LaneKeepingLongitudinalPolicy.get_input_space().n == 4


def test_policy_keep_accelerates_below_target(
    fake_policy: LaneKeepingLongitudinalPolicy,
    fake_vehicle: FakeVehicle,
) -> None:
    fake_vehicle.speed = 2.0
    steering, throttle = fake_policy._compute_action(fake_vehicle, DrivingAction.KEEP, 0.1)
    assert -1.0 <= steering <= 1.0
    assert 0.0 < throttle <= 1.0


def test_policy_stop_uses_emergency_brake(
    fake_policy: LaneKeepingLongitudinalPolicy,
    fake_vehicle: FakeVehicle,
) -> None:
    fake_vehicle.speed = 10.0
    _, throttle = fake_policy._compute_action(fake_vehicle, DrivingAction.STOP, 0.1)
    assert throttle == -1.0


def test_lateral_errors_steer_toward_lane(
    fake_policy: LaneKeepingLongitudinalPolicy,
    fake_vehicle: FakeVehicle,
) -> None:
    fake_vehicle.position = (5.0, 1.0)
    right_correction = fake_policy._compute_action(fake_vehicle, DrivingAction.KEEP, 0.1)[0]
    fake_policy.reset_controller_state()
    fake_vehicle.position = (5.0, -1.0)
    left_correction = fake_policy._compute_action(fake_vehicle, DrivingAction.KEEP, 0.1)[0]
    assert right_correction == -left_correction
    assert left_correction < 0.0 < right_correction


def test_heading_error_steers_toward_lane(
    fake_policy: LaneKeepingLongitudinalPolicy,
    fake_vehicle: FakeVehicle,
) -> None:
    fake_vehicle.heading_theta = 0.2
    assert fake_policy._compute_action(fake_vehicle, DrivingAction.KEEP, 0.1)[0] < 0.0


def test_missing_lane_fails_safe(
    fake_policy: LaneKeepingLongitudinalPolicy,
    fake_vehicle: FakeVehicle,
) -> None:
    fake_vehicle.navigation.current_lane = None
    fake_vehicle.lane = None
    assert fake_policy._compute_action(fake_vehicle, DrivingAction.KEEP, 0.1) == (0.0, -1.0)
    assert fake_policy.action_info["fail_safe"] is True


@pytest.mark.parametrize("action", [4, -1])
def test_invalid_action_fails_safe(
    fake_policy: LaneKeepingLongitudinalPolicy,
    fake_vehicle: FakeVehicle,
    action: int,
) -> None:
    assert fake_policy._compute_action(fake_vehicle, action, 0.1) == (0.0, -1.0)
    assert fake_policy.action_info["fail_safe_reason"] == "ValueError"


def test_non_finite_vehicle_speed_fails_safe(
    fake_policy: LaneKeepingLongitudinalPolicy,
    fake_vehicle: FakeVehicle,
) -> None:
    fake_vehicle.speed = math.nan
    assert fake_policy._compute_action(fake_vehicle, DrivingAction.KEEP, 0.1) == (0.0, -1.0)


def test_speed_limit_falls_back_to_vehicle_limit(
    fake_policy: LaneKeepingLongitudinalPolicy,
) -> None:
    vehicle = FakeVehicle(FakeLane(speed_limit=None))
    vehicle.speed = 2.0
    fake_policy._compute_action(vehicle, DrivingAction.KEEP, 0.1)
    assert fake_policy.action_info["target_speed_mps"] == 15.0


def test_reset_clears_all_controller_state(
    fake_policy: LaneKeepingLongitudinalPolicy,
    fake_vehicle: FakeVehicle,
) -> None:
    fake_policy._compute_action(fake_vehicle, DrivingAction.SLOW, 0.1)
    fake_policy.reset_controller_state()
    assert fake_policy._speed_pid.previous_error is None
    assert fake_policy._heading_pid.previous_error is None
    assert fake_policy._lateral_pid.previous_error is None


def test_action_info_contains_only_finite_runtime_numbers(
    fake_policy: LaneKeepingLongitudinalPolicy,
    fake_vehicle: FakeVehicle,
) -> None:
    fake_policy._compute_action(fake_vehicle, DrivingAction.SLOW, 0.1)

    def assert_finite(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                assert_finite(child)
        elif isinstance(value, list | tuple):
            for child in value:
                assert_finite(child)
        elif isinstance(value, float):
            assert math.isfinite(value)

    assert_finite(fake_policy.action_info)
