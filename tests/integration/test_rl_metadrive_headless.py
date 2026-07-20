import math
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from numpy.typing import NDArray
from stable_baselines3.common.vec_env import SubprocVecEnv

from mad_driving.config.loader import load_config
from mad_driving.config.models import AppConfig
from mad_driving.envs import MultiAgentSpeedEnv, create_control_metadrive_env
from mad_driving.interfaces import DecisionTrace
from mad_driving.training import run_training

SEED = 42
ACTION_SEQUENCE = tuple(index % 4 for index in range(100))


class CapturingSubprocVecEnv(SubprocVecEnv):
    latest: "CapturingSubprocVecEnv | None" = None

    def __init__(self, env_fns: list[Callable[[], gym.Env[Any, Any]]]) -> None:
        super().__init__(env_fns)
        type(self).latest = self


METADRIVE_INFO_KEYS = {
    "arrive_dest",
    "crash_vehicle",
    "crash_human",
    "out_of_road",
    "max_step",
}
EXPECTED_REWARD_COMPONENT_KEYS = (
    "progress_reward",
    "arrival_reward",
    "collision_penalty",
    "near_miss_penalty",
    "offroad_penalty",
    "rule_violation_penalty",
    "jerk_penalty",
    "unnecessary_brake_penalty",
    "standstill_penalty",
    "shield_intervention_penalty",
)


def assert_valid_observation(
    env: MultiAgentSpeedEnv,
    observation: NDArray[np.float32],
) -> None:
    assert observation.shape == (24,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert env.observation_space.contains(observation)


def run_trace_prefix() -> tuple[NDArray[np.float32], tuple[DecisionTrace, ...]]:
    env = MultiAgentSpeedEnv(
        load_config("configs/base.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        initial_observation, _ = env.reset(seed=SEED)
        traces = []
        for action in ACTION_SEQUENCE[:10]:
            _, _, terminated, truncated, info = env.step(action)
            trace = info["decision_trace"]
            assert isinstance(trace, DecisionTrace)
            traces.append(trace)
            assert not (terminated or truncated)
        return initial_observation.copy(), tuple(traces)
    finally:
        env.close()


def run_real_reset_sequence() -> tuple[tuple[int, int, int], ...]:
    env = MultiAgentSpeedEnv(
        load_config("configs/base.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        values = []
        _, info = env.reset(seed=SEED)
        values.append(
            (
                info["episode_rng_seed"],
                info["metadrive_scenario_index"],
                info["scenario_parameter_seed"],
            )
        )
        for _ in range(2):
            _, info = env.reset()
            values.append(
                (
                    info["episode_rng_seed"],
                    info["metadrive_scenario_index"],
                    info["scenario_parameter_seed"],
                )
            )
        return tuple(values)
    finally:
        env.close()


@pytest.mark.integration
def test_real_rl_environment_passes_gymnasium_checker() -> None:
    env = MultiAgentSpeedEnv(
        load_config("configs/base.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


@pytest.mark.integration
def test_real_control_adapter_rejects_out_of_range_before_simulator_side_effects() -> None:
    config = load_config("configs/base.yaml")
    metadrive_config = config.metadrive_dict()
    metadrive_config.update({"start_seed": 40, "num_scenarios": 3})
    env = create_control_metadrive_env(metadrive_config, config.control)
    try:
        assert env.engine is None
        for invalid_seed in (39, 43):
            with pytest.raises(ValueError, match="configured scenario range"):
                env.reset(seed=invalid_seed)
            assert env.engine is None

        for valid_seed in (40, 42):
            _, info = env.reset(seed=valid_seed)
            assert info["env_seed"] == valid_seed
            assert info["metadrive_scenario_index"] == valid_seed
            assert env.current_seed == valid_seed
    finally:
        env.close()


@pytest.mark.integration
def test_real_rl_environment_runs_100_finite_headless_steps() -> None:
    env = MultiAgentSpeedEnv(
        load_config("configs/base.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        initial_observation, _ = env.reset(seed=SEED)
        assert_valid_observation(env, initial_observation)

        steps_completed = 0
        for action in ACTION_SEQUENCE:
            result = env.step(action)
            assert len(result) == 5
            observation, reward, terminated, truncated, info = result
            assert_valid_observation(env, observation)
            assert math.isfinite(reward)
            assert METADRIVE_INFO_KEYS <= info.keys()

            reward_components = info["reward_components"]
            assert isinstance(reward_components, dict)
            assert tuple(reward_components) == EXPECTED_REWARD_COMPONENT_KEYS
            assert all(
                isinstance(value, int | float) and math.isfinite(value)
                for value in reward_components.values()
            )
            trace = info["decision_trace"]
            assert isinstance(trace, DecisionTrace)
            assert trace.reward_components == reward_components
            assert all(math.isfinite(value) for value in trace.reward_components.values())

            steps_completed += 1
            if terminated or truncated:
                break

        assert steps_completed == len(ACTION_SEQUENCE)
    finally:
        env.close()


@pytest.mark.integration
def test_real_rl_environment_reports_the_actual_allocated_scenario_identity() -> None:
    env = MultiAgentSpeedEnv(
        load_config("configs/base.yaml"),
        role="train",
        worker_index=4,
    )
    try:
        _, info = env.reset(seed=SEED)
        assert info["episode_rng_seed"] == SEED
        assert info["simulator_seed"] == info["metadrive_scenario_index"]
        assert 0 <= info["metadrive_scenario_index"] < 10_000
        assert 0 <= info["scenario_parameter_seed"] < 10_000
        assert info["scenario_seed"] == info["scenario_parameter_seed"]
        assert info["role"] == "train"
        assert info["worker_index"] == 4
    finally:
        env.close()


@pytest.mark.integration
def test_real_validation_environment_uses_actual_validation_scenario_identity() -> None:
    config = load_config("configs/base.yaml")
    created = []

    def factory(options: dict[str, object], control: Any):
        environment = create_control_metadrive_env(options, control)
        created.append(environment)
        return environment

    env = MultiAgentSpeedEnv(
        config,
        role="validation",
        worker_index=0,
        env_factory=factory,
    )
    try:
        _, info = env.reset(seed=SEED)
        actual_index = info["metadrive_scenario_index"]
        assert 10_000 <= actual_index < 11_000
        assert info["env_seed"] == actual_index
        assert created[0].current_seed == actual_index
        assert info["role"] == "validation"
        assert info["worker_index"] == 0
    finally:
        env.close()


@pytest.mark.integration
def test_real_rl_implicit_reset_sequence_advances_and_is_reproducible() -> None:
    first = run_real_reset_sequence()
    second = run_real_reset_sequence()

    assert first == second
    assert len({episode_rng_seed for episode_rng_seed, _, _ in first}) == len(first)
    assert all(0 <= simulator_seed < 10_000 for _, simulator_seed, _ in first)
    assert all(0 <= scenario_seed < 10_000 for _, _, scenario_seed in first)


@pytest.mark.integration
def test_real_rl_environment_repeats_same_seed_initial_observation_and_traces() -> None:
    first_observation, first_traces = run_trace_prefix()
    second_observation, second_traces = run_trace_prefix()

    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_traces == second_traces


@pytest.mark.integration
def test_real_single_metadrive_training_isolates_evaluation_engine(tmp_path: Path) -> None:
    payload = load_config("configs/train.yaml").model_dump(mode="python")
    payload["metadrive"]["horizon"] = 8
    payload["training"].update(
        {
            "n_steps": 8,
            "batch_size": 8,
            "n_epochs": 1,
            "smoke_timesteps": 8,
            "checkpoint_interval_steps": 8,
            "eval_interval_steps": 8,
            "eval_episodes": 1,
        }
    )
    config = AppConfig.model_validate(payload)

    CapturingSubprocVecEnv.latest = None
    result = run_training(
        config,
        smoke=True,
        run_dir=tmp_path / "real-training",
        subproc_vec_env_factory=CapturingSubprocVecEnv,
    )

    assert result.timesteps == 8
    assert zipfile.is_zipfile(result.best_checkpoint)
    assert zipfile.is_zipfile(result.final_checkpoint)
    evaluation = CapturingSubprocVecEnv.latest
    assert evaluation is not None
    assert evaluation.closed is True
    assert all(not process.is_alive() for process in evaluation.processes)
