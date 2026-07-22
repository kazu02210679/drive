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


def run_cut_in_prefix() -> tuple[tuple[tuple[float, float], ...], float, float]:
    environment = MultiAgentSpeedEnv(
        load_config("configs/base.yaml", "configs/scenarios/cut_in.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        observation, info = environment.reset(seed=42)
        assert info["scenario_id"] == "cut_in"
        manager = environment._environment.engine.scenario_actor_manager
        assert manager.actor_ids() == ("cut-in",)
        ego_lane = environment._environment.vehicle.navigation.current_lane
        _, initial_lateral_m = ego_lane.local_coordinates(
            manager.actor_state("cut-in").position_xy_m
        )
        assert abs(initial_lateral_m) > 0.1
        parameters = info["scenario_parameters"]
        trigger_step = parameters["trigger_step"]
        merge_steps = parameters["merge_steps"]
        assert isinstance(trigger_step, int)
        assert isinstance(merge_steps, int)
        positions: list[tuple[float, float]] = []
        final_lateral_m = initial_lateral_m
        for _step_index in range(1, trigger_step + merge_steps + 1):
            observation, reward, terminated, truncated, _ = environment.step(3)
            assert np.isfinite(observation).all()
            assert math.isfinite(reward)
            assert not (terminated or truncated)
            state = manager.actor_state("cut-in")
            positions.append(state.position_xy_m)
            _, final_lateral_m = ego_lane.local_coordinates(state.position_xy_m)
        assert len(positions) >= 20
        return tuple(positions[:20]), initial_lateral_m, final_lateral_m
    finally:
        environment.close()


@pytest.mark.integration
def test_real_cut_in_is_deterministic_merges_and_cleans_up() -> None:
    first_positions, first_initial_lateral, first_final_lateral = run_cut_in_prefix()
    second_positions, second_initial_lateral, second_final_lateral = run_cut_in_prefix()

    assert first_positions == second_positions
    assert abs(first_initial_lateral) > 0.1
    assert first_final_lateral == pytest.approx(0.0, abs=0.1)
    assert second_initial_lateral == pytest.approx(first_initial_lateral)
    assert second_final_lateral == pytest.approx(first_final_lateral)

    environment = MultiAgentSpeedEnv(
        load_config("configs/base.yaml", "configs/scenarios/cut_in.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        environment.reset(seed=42)
        manager = environment._environment.engine.scenario_actor_manager
        previous_actor = manager.engine.get_objects(["cut-in"])["cut-in"]
        environment.reset(seed=43)

        assert previous_actor not in manager.engine.get_objects().values()
    finally:
        environment.close()
