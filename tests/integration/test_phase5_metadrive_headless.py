import math
from itertools import pairwise

import numpy as np
import pytest

from mad_driving.config.loader import load_config
from mad_driving.envs import MultiAgentSpeedEnv


def run_prefix() -> tuple[np.ndarray, tuple[float, ...], tuple[float, ...]]:
    environment = MultiAgentSpeedEnv(
        load_config("configs/base.yaml", "configs/scenarios/lead_brake.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        observation, info = environment.reset(seed=42)
        assert info["scenario_id"] == "lead_brake"
        assert info["difficulty_level"] == 1
        manager = environment._environment.engine.scenario_actor_manager
        assert manager.actor_ids() == ("lead-brake",)
        parameters = info["scenario_parameters"]
        trigger_step = parameters["trigger_step"]
        assert isinstance(trigger_step, int)
        rewards = []
        braking_speeds = []
        for step_index in range(1, trigger_step + 4):
            observation, reward, terminated, truncated, _ = environment.step(0)
            assert np.isfinite(observation).all()
            assert math.isfinite(reward)
            rewards.append(reward)
            assert not (terminated or truncated)
            if step_index >= trigger_step:
                braking_speeds.append(manager.actor_state("lead-brake").velocity_xy_mps[0])
        assert len(braking_speeds) >= 4
        assert sum(
            later < earlier for earlier, later in pairwise(braking_speeds)
        ) >= 2
        return observation.copy(), tuple(rewards), tuple(braking_speeds)
    finally:
        environment.close()


@pytest.mark.integration
def test_real_lead_brake_actor_is_deterministic_and_does_not_leak() -> None:
    first_observation, first_rewards, first_braking_speeds = run_prefix()
    second_observation, second_rewards, second_braking_speeds = run_prefix()

    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_rewards == second_rewards
    assert first_braking_speeds == second_braking_speeds

    environment = MultiAgentSpeedEnv(
        load_config("configs/base.yaml", "configs/scenarios/lead_brake.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        environment.reset(seed=42)
        manager = environment._environment.engine.scenario_actor_manager
        assert manager.actor_ids() == ("lead-brake",)
        environment.reset(seed=43)
        assert manager.actor_ids() == ("lead-brake",)
    finally:
        environment.close()
