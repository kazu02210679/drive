from typing import Any

import pytest
from pydantic import ValidationError

from mad_driving.config.models import (
    AppConfig,
    FloatRangeConfig,
    ObservationConfig,
    PPOConfig,
    RewardConfig,
)


def minimum_app_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "scenario_id": "phase4_config",
        "decision_steps": 10,
        "fixed_action": [0.0, 0.25],
        "metadrive": {},
    }


def test_phase4_defaults_match_specification() -> None:
    config = AppConfig.model_validate(minimum_app_config())

    assert config.observation.max_ttc_s == 10.0
    assert config.reward.progress_per_meter == 0.10
    assert config.reward.collision_crossing_actor == 500.0
    assert config.training.algorithm == "PPO"
    assert config.training.n_steps == 2048
    assert config.training.smoke_timesteps == 5_000
    assert config.training.total_timesteps == 500_000


def test_simulation_timing_and_seed_split_defaults() -> None:
    config = AppConfig.model_validate(minimum_app_config())

    assert config.metadrive.physics_dt_s == 0.02
    assert config.metadrive.decision_repeat == 5
    assert config.metadrive.decision_dt_s == 0.10
    assert config.metadrive.lane_width_m == 3.5
    assert config.scenarios.train.range == range(0, 10_000)
    assert config.scenarios.validation.range == range(10_000, 11_000)
    assert config.scenarios.test.range == range(20_000, 21_000)
    assert config.metadrive_dict() == {
        "use_render": False,
        "image_observation": False,
        "num_scenarios": 1,
        "start_seed": 0,
        "traffic_density": 0.1,
        "horizon": 200,
        "physics_world_step_size": 0.02,
        "decision_repeat": 5,
        "map_config": {"lane_width": 3.5},
    }


def test_decision_dt_must_equal_physics_dt_times_repeat() -> None:
    payload = minimum_app_config()
    payload["metadrive"] = {
        "physics_dt_s": 0.02,
        "decision_repeat": 5,
        "decision_dt_s": 0.2,
    }

    with pytest.raises(ValidationError, match="decision_dt_s"):
        AppConfig.model_validate(payload)


def test_scenario_seed_ranges_must_not_overlap() -> None:
    payload = minimum_app_config()
    payload["scenarios"] = {
        "train": {"seed_start": 0, "seed_count": 100},
        "validation": {"seed_start": 50, "seed_count": 10},
        "test": {"seed_start": 200, "seed_count": 10},
    }

    with pytest.raises(ValidationError, match="overlap"):
        AppConfig.model_validate(payload)


def test_float_range_requires_ordered_finite_bounds() -> None:
    with pytest.raises(ValidationError, match="minimum"):
        FloatRangeConfig(minimum=2.0, maximum=1.0)


def test_phase4_explicit_values_are_preserved_in_app_config() -> None:
    payload = minimum_app_config()
    payload.update(
        {
            "observation": {"max_speed_mps": 32.5, "max_ttc_s": 7.5},
            "reward": {"progress_per_meter": 0.25, "shield_intervention": 4.0},
            "training": {
                "learning_rate": 0.001,
                "n_steps": 128,
                "batch_size": 32,
                "num_envs": 2,
                "run_root": "custom-runs",
            },
        }
    )

    config = AppConfig.model_validate(payload)

    assert config.observation.max_speed_mps == 32.5
    assert config.observation.max_ttc_s == 7.5
    assert config.reward.progress_per_meter == 0.25
    assert config.reward.shield_intervention == 4.0
    assert config.training.learning_rate == 0.001
    assert config.training.n_steps == 128
    assert config.training.batch_size == 32
    assert config.training.num_envs == 2
    assert config.training.run_root == "custom-runs"


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (ObservationConfig, {"max_ttc_s": "10.0"}),
        (RewardConfig, {"near_miss_max": float("nan")}),
        (PPOConfig, {"batch_size": 0}),
        (PPOConfig, {"algorithm": "DQN"}),
    ],
)
def test_phase4_models_reject_invalid_or_coerced_values(
    model: type[Any], payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_ppo_rollout_size_must_be_divisible_by_batch() -> None:
    with pytest.raises(ValidationError, match="batch_size"):
        PPOConfig(n_steps=10, num_envs=1, batch_size=6)


def test_standstill_speed_threshold_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="standstill_speed_mps"):
        RewardConfig(standstill_speed_mps=0.0)


def test_stale_unnecessary_brake_lookahead_setting_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unnecessary_brake_lookahead_steps"):
        RewardConfig.model_validate({"unnecessary_brake_lookahead_steps": 3})
