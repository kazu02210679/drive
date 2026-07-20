import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from numpy.typing import NDArray

from mad_driving.config.loader import load_config
from mad_driving.envs import MultiAgentSpeedEnv
from mad_driving.interfaces import DecisionTrace

SEED = 42
ACTION_SEQUENCE = tuple(index % 4 for index in range(100))
METADRIVE_INFO_KEYS = {
    "arrive_dest",
    "crash_vehicle",
    "crash_human",
    "out_of_road",
    "max_step",
}


def assert_valid_observation(
    env: MultiAgentSpeedEnv,
    observation: NDArray[np.float32],
) -> None:
    assert observation.shape == (24,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert env.observation_space.contains(observation)


def run_trace_prefix() -> tuple[NDArray[np.float32], tuple[DecisionTrace, ...]]:
    env = MultiAgentSpeedEnv(load_config("configs/base.yaml"))
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


@pytest.mark.integration
def test_real_rl_environment_passes_gymnasium_checker() -> None:
    env = MultiAgentSpeedEnv(load_config("configs/base.yaml"))
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


@pytest.mark.integration
def test_real_rl_environment_runs_100_finite_headless_steps() -> None:
    env = MultiAgentSpeedEnv(load_config("configs/base.yaml"))
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
def test_real_rl_environment_repeats_same_seed_initial_observation_and_traces() -> None:
    first_observation, first_traces = run_trace_prefix()
    second_observation, second_traces = run_trace_prefix()

    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_traces == second_traces
