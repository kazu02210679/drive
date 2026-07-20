import math
from typing import Any

import pytest
from pydantic import ValidationError

from mad_driving.config.models import (
    AppConfig,
    ControlConfig,
    CoordinatorConfig,
    ShieldConfig,
    SpeedPIDConfig,
    SteeringPIDConfig,
)


def minimum_app_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "scenario_id": "phase3_config",
        "decision_steps": 10,
        "fixed_action": [0.0, 0.25],
        "metadrive": {},
    }


def test_phase3_defaults_are_available_to_old_configs() -> None:
    config = AppConfig.model_validate(minimum_app_config())

    assert config.coordinator.conflict_min_action == 1
    assert config.coordinator.severe_min_action == 2
    assert config.coordinator.severe_threshold == 0.75
    assert config.shield.mode == "enforce"
    assert config.shield.imminent_ttc_s == 1.0
    assert config.shield.caution_ttc_s == 3.0
    assert config.control.speed.emergency_deceleration_mps2 == -6.0
    assert config.control.steering.lookahead_m == 1.0


def test_phase3_models_are_strict_and_frozen() -> None:
    config = CoordinatorConfig()

    with pytest.raises(ValidationError, match="frozen"):
        config.severe_threshold = 0.5  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CoordinatorConfig.model_validate({"unknown": 1})


@pytest.mark.parametrize(
    "payload",
    [
        {"imminent_ttc_s": 4.0, "caution_ttc_s": 3.0},
        {"emergency_margin_m": 6.0, "caution_margin_m": 5.0},
        {"mode": "unknown"},
        {"imminent_ttc_s": math.nan},
        {"missing_agent_action": 4},
        {"multiple_missing_action": -1},
    ],
)
def test_shield_config_rejects_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ShieldConfig.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"conflict_min_action": -1},
        {"severe_min_action": 4},
        {"severe_threshold": math.inf},
        {"severe_threshold": 1.1},
    ],
)
def test_coordinator_config_rejects_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CoordinatorConfig.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "normal_deceleration_mps2": -7.0,
            "emergency_deceleration_mps2": -6.0,
        },
        {"normal_deceleration_mps2": 0.0},
        {"emergency_deceleration_mps2": 0.0},
        {"max_acceleration_mps2": 0.0},
        {"kp": -0.1},
        {"integral_limit": math.nan},
    ],
)
def test_speed_pid_config_rejects_invalid_values(payload: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        SpeedPIDConfig.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"heading_kp": -0.1},
        {"lateral_kd": math.inf},
        {"integral_limit": 0.0},
        {"lookahead_m": 0.0},
    ],
)
def test_steering_pid_config_rejects_invalid_values(payload: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        SteeringPIDConfig.model_validate(payload)


def test_explicit_phase3_config_is_nested_under_app_config() -> None:
    payload = minimum_app_config()
    payload.update(
        {
            "coordinator": {"severe_threshold": 0.8},
            "shield": {"mode": "monitor", "caution_ttc_s": 4.0},
            "control": {
                "speed": {"kp": 0.6},
                "steering": {"lookahead_m": 2.0},
            },
        }
    )

    config = AppConfig.model_validate(payload)

    assert config.coordinator.severe_threshold == 0.8
    assert config.shield.mode == "monitor"
    assert config.shield.caution_ttc_s == 4.0
    assert config.control.speed.kp == 0.6
    assert config.control.steering.lookahead_m == 2.0
    assert isinstance(config.control, ControlConfig)
