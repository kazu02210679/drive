import zipfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import yaml
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3 import PPO

from mad_driving.config.models import AppConfig
from mad_driving.training import run_training


class TinyDeterministicEnv(gym.Env[NDArray[np.float32], int]):
    """Fast deterministic 24D/Discrete(4) environment for real PPO integration."""

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
    action_space = spaces.Discrete(4)

    def __init__(self) -> None:
        self.steps = 0
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.steps = 0
        return np.zeros(24, dtype=np.float32), {}

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        self.steps += 1
        observation = np.full(24, self.steps / 4.0, dtype=np.float32)
        reward = float(action == self.steps % 4)
        return observation, reward, self.steps == 4, False, {}

    def close(self) -> None:
        self.closed = True


def make_real_ppo_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "seed": 7,
            "scenario_id": "tiny-ppo-artifacts",
            "decision_steps": 4,
            "fixed_action": [0.0, 0.0],
            "metadrive": {"use_render": False},
            "training": {
                "n_steps": 8,
                "batch_size": 8,
                "total_timesteps": 16,
                "checkpoint_interval_steps": 8,
                "eval_interval_steps": 8,
                "eval_episodes": 1,
            },
        }
    )


def assert_loadable_policy(checkpoint: Path) -> None:
    model = PPO.load(checkpoint, device="cpu")
    action, _ = model.predict(np.zeros(24, dtype=np.float32), deterministic=True)
    predicted_action = int(np.asarray(action).item())
    assert 0 <= predicted_action <= 3


@pytest.mark.integration
def test_real_ppo_writes_artifacts_and_resumes_transactionally(tmp_path: Path) -> None:
    config = make_real_ppo_config()
    run_dir = tmp_path / "run"
    environments: list[TinyDeterministicEnv] = []

    def env_factory(received_config: AppConfig) -> TinyDeterministicEnv:
        assert received_config is config
        environment = TinyDeterministicEnv()
        environments.append(environment)
        return environment

    first_result = run_training(
        config,
        smoke=False,
        run_dir=run_dir,
        env_factory=env_factory,
    )

    checkpoints_dir = run_dir / "checkpoints"
    periodic_checkpoints = sorted(checkpoints_dir.glob("ppo_checkpoint_*_steps.zip"))
    assert {path.name for path in periodic_checkpoints} == {
        "ppo_checkpoint_8_steps.zip",
        "ppo_checkpoint_16_steps.zip",
    }
    all_checkpoints = [
        *periodic_checkpoints,
        first_result.best_checkpoint,
        first_result.final_checkpoint,
    ]
    assert first_result.timesteps == 16
    assert all(path.is_file() and zipfile.is_zipfile(path) for path in all_checkpoints)
    assert_loadable_policy(first_result.final_checkpoint)
    assert list((run_dir / "tensorboard").rglob("events.out.tfevents.*"))

    resolved_config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved_config == config.model_dump(mode="json")
    assert {
        key: resolved_config["training"][key]
        for key in (
            "n_steps",
            "batch_size",
            "total_timesteps",
            "checkpoint_interval_steps",
            "eval_interval_steps",
        )
    } == {
        "n_steps": 8,
        "batch_size": 8,
        "total_timesteps": 16,
        "checkpoint_interval_steps": 8,
        "eval_interval_steps": 8,
    }
    assert len(environments) == 2
    assert all(environment.closed for environment in environments)

    resumed_result = run_training(
        config,
        smoke=False,
        run_dir=run_dir,
        resume_from=first_result.final_checkpoint,
        env_factory=env_factory,
    )

    assert resumed_result.final_checkpoint == first_result.final_checkpoint
    assert resumed_result.timesteps == 16
    assert zipfile.is_zipfile(resumed_result.final_checkpoint)
    assert_loadable_policy(resumed_result.final_checkpoint)
    assert all(
        zipfile.is_zipfile(path)
        for path in [
            *checkpoints_dir.glob("ppo_checkpoint_*_steps.zip"),
            resumed_result.best_checkpoint,
            resumed_result.final_checkpoint,
        ]
    )
    assert not list(checkpoints_dir.glob(".training-*"))
    assert len(environments) == 4
    assert all(environment.closed for environment in environments)
