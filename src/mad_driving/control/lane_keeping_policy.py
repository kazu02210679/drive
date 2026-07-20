"""MetaDrive Policy for lane keeping and high-level longitudinal actions."""

from math import atan2, cos, sin
from typing import Any

import gymnasium as gym
import numpy as np
from metadrive.policy.base_policy import BasePolicy  # type: ignore[import-untyped]

from mad_driving.config.models import ControlConfig
from mad_driving.control.action_mapper import target_speed_mps
from mad_driving.control.actions import DrivingAction
from mad_driving.control.pid import BoundedPID
from mad_driving.world_model.validation import decision_interval_s


class LaneKeepingLongitudinalPolicy(BasePolicy):  # type: ignore[misc]
    """Translate four discrete decisions into normalized vehicle controls."""

    def __init__(self, control_object: Any, random_seed: int | None = None) -> None:
        super().__init__(control_object=control_object, random_seed=random_seed)
        control_payload = self.engine.global_config["control_config"]
        if hasattr(control_payload, "get_dict"):
            control_payload = control_payload.get_dict()
        self._control_config = ControlConfig.model_validate(control_payload)
        self._decision_interval_s = decision_interval_s(self.engine.global_config)
        self._build_controllers()

    @classmethod
    def get_input_space(cls) -> gym.spaces.Discrete[np.int64]:
        """Expose the four high-level actions to Gymnasium."""

        return gym.spaces.Discrete(4)

    def act(self, agent_id: str) -> list[float]:
        """Read one external action and return steering plus throttle/brake."""

        raw_action = self.engine.external_actions[agent_id]
        steering, throttle = self._compute_action(
            self.control_object,
            raw_action,
            self._decision_interval_s,
        )
        return [steering, throttle]

    def reset(self) -> None:
        """Clear Policy diagnostics and all controller state."""

        super().reset()
        self.reset_controller_state()

    def _build_controllers(self) -> None:
        speed = self._control_config.speed
        steering = self._control_config.steering
        self._speed_pid = BoundedPID(
            speed.kp,
            speed.ki,
            speed.kd,
            speed.integral_limit,
        )
        self._heading_pid = BoundedPID(
            steering.heading_kp,
            steering.heading_ki,
            steering.heading_kd,
            steering.integral_limit,
        )
        self._lateral_pid = BoundedPID(
            steering.lateral_kp,
            steering.lateral_ki,
            steering.lateral_kd,
            steering.integral_limit,
        )

    def reset_controller_state(self) -> None:
        """Reset speed, heading, and lateral PID state."""

        self._speed_pid.reset()
        self._heading_pid.reset()
        self._lateral_pid.reset()

    def _compute_action(
        self,
        vehicle: Any,
        action_value: DrivingAction | int,
        dt_s: float,
    ) -> tuple[float, float]:
        try:
            action = DrivingAction(action_value)
            steering, throttle, target = self._calculate_action(vehicle, action, dt_s)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            self.action_info = {
                "action": [0.0, -1.0],
                "fail_safe": True,
                "fail_safe_reason": type(exc).__name__,
            }
            return 0.0, -1.0
        self.action_info = {
            "action": [steering, throttle],
            "requested_action": int(action),
            "target_speed_mps": target,
            "steering": steering,
            "throttle_brake": throttle,
            "fail_safe": False,
            "fail_safe_reason": None,
        }
        return steering, throttle

    def _calculate_action(
        self,
        vehicle: Any,
        action: DrivingAction,
        dt_s: float,
    ) -> tuple[float, float, float]:
        navigation = getattr(vehicle, "navigation", None)
        lane = (
            getattr(navigation, "current_lane", None)
            if navigation is not None
            else None
        )
        if lane is None:
            lane = getattr(vehicle, "lane", None)
        if lane is None:
            raise ValueError("lane is unavailable")

        position = tuple(float(value) for value in vehicle.position)
        longitudinal, lateral = lane.local_coordinates(position)
        lookahead = self._control_config.steering.lookahead_m
        lane_heading = float(lane.heading_theta_at(longitudinal + lookahead))
        vehicle_heading = float(vehicle.heading_theta)
        current_speed = float(vehicle.speed)
        speed_limit = self._speed_limit_mps(vehicle, lane)
        target = target_speed_mps(action, current_speed, speed_limit)

        heading_error = atan2(
            sin(vehicle_heading - lane_heading),
            cos(vehicle_heading - lane_heading),
        )
        heading_command = self._heading_pid.update(
            heading_error,
            dt_s,
            -1.0,
            1.0,
        )
        lateral_command = self._lateral_pid.update(
            -float(lateral),
            dt_s,
            -1.0,
            1.0,
        )
        steering = min(max(heading_command + lateral_command, -1.0), 1.0)

        speed_config = self._control_config.speed
        lower = (
            speed_config.emergency_deceleration_mps2
            if action is DrivingAction.STOP
            else speed_config.normal_deceleration_mps2
        )
        if action is DrivingAction.STOP:
            self._speed_pid.reset()
            desired_acceleration = lower
        else:
            desired_acceleration = self._speed_pid.update(
                target - current_speed,
                dt_s,
                lower,
                speed_config.max_acceleration_mps2,
            )
        throttle = (
            desired_acceleration / speed_config.max_acceleration_mps2
            if desired_acceleration >= 0.0
            else desired_acceleration / abs(lower)
        )
        return steering, min(max(throttle, -1.0), 1.0), target

    @staticmethod
    def _speed_limit_mps(vehicle: Any, lane: Any) -> float:
        if hasattr(lane, "speed_limit"):
            return float(lane.speed_limit) / 3.6
        return float(vehicle.max_speed_m_s)
