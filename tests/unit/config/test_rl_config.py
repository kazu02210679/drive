from typing import Any

import pytest
from pydantic import ValidationError

from mad_driving.config.models import AppConfig, ObservationConfig, PPOConfig, RewardConfig


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
