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
