import math
from dataclasses import asdict
from typing import Any

import pytest

from mad_driving.cli.control_smoke import run_control_smoke
from mad_driving.config.loader import load_config
from mad_driving.control import DrivingAction
from mad_driving.envs.control_metadrive_env import create_control_metadrive_env
from mad_driving.interfaces import DecisionTrace


def assert_finite_tree(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            assert_finite_tree(child)
    elif isinstance(value, tuple | list):
        for child in value:
            assert_finite_tree(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


@pytest.mark.integration
def test_control_policy_exposes_discrete_four_and_steps_headless() -> None:
    config = load_config("configs/base.yaml")
    env = create_control_metadrive_env(config.metadrive_dict(), config.control)
    try:
        env.reset(seed=config.seed)
        assert env.action_space.n == 4
        for action in DrivingAction:
            result = env.step(int(action))
            assert len(result) == 5
            assert math.isfinite(env.agent.steering)
            assert math.isfinite(env.agent.throttle_brake)
    finally:
        env.close()


@pytest.mark.integration
def test_real_policy_stop_reduces_speed() -> None:
    config = load_config("configs/base.yaml")
    env = create_control_metadrive_env(config.metadrive_dict(), config.control)
    try:
        env.reset(seed=config.seed)
        keep_samples = 0
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(int(DrivingAction.KEEP))
            keep_samples += 1
            if terminated or truncated:
                break
        assert keep_samples == 20
        speed_before_stop = env.agent.speed

        stop_samples = 0
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(int(DrivingAction.STOP))
            stop_samples += 1
            if terminated or truncated:
                break
        assert stop_samples == 20
        assert speed_before_stop > 0.0
        assert env.agent.speed < speed_before_stop
    finally:
        env.close()


@pytest.mark.integration
def test_real_policy_does_not_steer_out_of_road() -> None:
    config = load_config("configs/base.yaml")
    env = create_control_metadrive_env(config.metadrive_dict(), config.control)
    try:
        env.reset(seed=config.seed)
        for _ in range(60):
            _, _, terminated, truncated, info = env.step(int(DrivingAction.KEEP))
            assert info["out_of_road"] is False
            if terminated or truncated:
                break
    finally:
        env.close()


@pytest.mark.integration
def test_real_complete_control_pipeline_runs_100_steps_and_closes() -> None:
    config = load_config("configs/base.yaml")
    created = []

    def factory(options: dict[str, object], control: Any):
        env = create_control_metadrive_env(options, control)
        created.append(env)
        return env

    result = run_control_smoke(config, env_factory=factory)

    assert result.steps_completed == 100
    assert result.terminated is False
    assert result.truncated is False
    assert sum(result.action_counts) == result.steps_completed
    assert isinstance(result.final_trace, DecisionTrace)
    assert_finite_tree(asdict(result))
    assert created[0].engine is None
