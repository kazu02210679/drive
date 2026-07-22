import json
import math
from itertools import pairwise

import numpy as np
import pytest
from metadrive.component.traffic_participants.cyclist import Cyclist

from mad_driving.config.loader import load_config
from mad_driving.envs import MultiAgentSpeedEnv
from mad_driving.interfaces import DecisionTrace

ALLOWED_SCENARIOS_BY_LEVEL = {
    0: {"nominal"},
    1: {"lead_brake"},
    2: {"lead_brake", "cut_in"},
    3: {"occluded_crossing"},
}


def fixed_level_config(level: int):
    payload = load_config("configs/base.yaml").model_dump(mode="python")
    payload["scenario_id"] = "phase5"
    payload["scenarios"]["selection"] = "auto"
    payload["scenarios"]["curriculum"] = {
        "mode": "fixed",
        "fixed_level": level,
    }
    return type(load_config("configs/base.yaml")).model_validate(payload)


def run_fixed_level_replay(level: int) -> bytes:
    environment = MultiAgentSpeedEnv(
        fixed_level_config(level),
        role="train",
        worker_index=0,
    )
    try:
        observation, reset_info = environment.reset(seed=42)
        assert observation.shape == (24,)
        assert np.isfinite(observation).all()
        manager = environment._environment.engine.scenario_actor_manager
        trajectory: list[dict[str, object]] = []
        outcomes: list[dict[str, bool]] = []
        trace_metadata: list[dict[str, object]] = []
        for _ in range(3):
            observation, reward, terminated, truncated, info = environment.step(3)
            assert observation.shape == (24,)
            assert np.isfinite(observation).all()
            assert math.isfinite(reward)
            trajectory.append(
                {
                    actor_id: {
                        "position_xy_m": list(manager.actor_state(actor_id).position_xy_m),
                        "velocity_xy_mps": list(manager.actor_state(actor_id).velocity_xy_mps),
                    }
                    for actor_id in manager.actor_ids()
                }
            )
            outcomes.append(
                {
                    "scenario_success": info["scenario_success"],
                    "scenario_failure": info["scenario_failure"],
                    "collision_occurred": info["collision_occurred"],
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            trace = info["decision_trace"]
            assert isinstance(trace, DecisionTrace)
            trace_metadata.append(
                {
                    "environment_seed": trace.episode_rng_seed,
                    "scenario_selection_seed": trace.metadrive_scenario_index,
                    "scenario_parameter_seed": trace.scenario_parameter_seed,
                    "scenario_id": trace.scenario_id,
                    "difficulty_level": trace.difficulty_level,
                    "role": trace.role,
                    "worker_index": trace.worker_index,
                }
            )
            if terminated or truncated:
                break
        return json.dumps(
            {
                "scenario_id": reset_info["scenario_id"],
                "difficulty_level": reset_info["difficulty_level"],
                "scenario_parameters": reset_info["scenario_parameters"],
                "trajectory": trajectory,
                "outcomes": outcomes,
                "trace_metadata": trace_metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    finally:
        environment.close()


@pytest.mark.integration
@pytest.mark.parametrize("level", range(4))
def test_real_fixed_levels_select_only_allowed_scenarios_and_replay_byte_for_byte(
    level: int,
) -> None:
    first = run_fixed_level_replay(level)
    second = run_fixed_level_replay(level)

    assert first == second
    payload = json.loads(first)
    assert payload["scenario_id"] in ALLOWED_SCENARIOS_BY_LEVEL[level]
    assert payload["difficulty_level"] == level
    assert all(item["scenario_id"] == payload["scenario_id"] for item in payload["trace_metadata"])
    assert all(item["difficulty_level"] == level for item in payload["trace_metadata"])


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
        assert environment._environment.scenario_ego_collided_with("cut-in") is False
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


def run_occluded_crossing_prefix() -> tuple[tuple[tuple[float, float], ...], bool]:
    environment = MultiAgentSpeedEnv(
        load_config("configs/base.yaml", "configs/scenarios/occluded_crossing.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        _observation, info = environment.reset(seed=42)
        assert info["scenario_id"] == "occluded_crossing"
        manager = environment._environment.engine.scenario_actor_manager
        assert manager.actor_ids() == ("crossing-cyclist", "static-occluder", "crossing-lead")
        cyclist = manager.engine.get_objects(["crossing-cyclist"])["crossing-cyclist"]
        assert cyclist.id == "crossing-cyclist"
        assert isinstance(cyclist, Cyclist)

        reset_frame = environment._frame
        assert reset_frame is not None
        assert "crossing-cyclist" not in {
            actor.actor_id for actor in reset_frame.observation.visible_actors
        }
        reset_visible_ids = {actor.actor_id for actor in reset_frame.observation.visible_actors}
        assert "crossing-lead" in reset_visible_ids
        privileged = {actor.actor_id: actor for actor in reset_frame.privileged.all_actors}
        assert privileged["crossing-cyclist"].visible is False
        assert privileged["crossing-cyclist"].occluded is True

        _observation, reward, terminated, truncated, _info = environment.step(0)
        assert math.isfinite(reward)
        assert not (terminated or truncated)
        moving_frame = environment._frame
        assert moving_frame is not None
        assert moving_frame.privileged.minimum_actual_ttc_s is not None
        assert math.isfinite(moving_frame.privileged.minimum_actual_ttc_s)

        parameters = info["scenario_parameters"]
        trigger_step = parameters["trigger_step"]
        crossing_start_offset_m = parameters["crossing_start_offset_m"]
        crossing_speed_mps = parameters["crossing_speed_mps"]
        reveal_lateral_m = parameters["reveal_lateral_m"]
        clear_lateral_m = parameters["clear_lateral_m"]
        survival_s = parameters["survival_s"]
        lead_speed_mps = parameters["lead_speed_mps"]
        assert isinstance(trigger_step, int)
        assert isinstance(crossing_start_offset_m, float)
        assert isinstance(crossing_speed_mps, float)
        assert isinstance(reveal_lateral_m, float)
        assert isinstance(clear_lateral_m, float)
        assert isinstance(survival_s, float)
        assert isinstance(lead_speed_mps, float)
        clear_steps = math.ceil(
            (crossing_start_offset_m + clear_lateral_m) / crossing_speed_mps / 0.1
        )
        survival_steps = math.ceil(survival_s / 0.1)
        positions: list[tuple[float, float]] = []
        revealed = False
        cleared = False
        completed = False
        for _ in range(1, trigger_step + clear_steps + survival_steps + 6):
            _observation, reward, terminated, truncated, step_info = environment.step(3)
            assert math.isfinite(reward)
            positions.append(manager.actor_state("crossing-cyclist").position_xy_m)
            frame = environment._frame
            assert frame is not None
            actor_ids = {actor.actor_id for actor in frame.observation.visible_actors}
            lead_state = manager.actor_state("crossing-lead")
            assert math.hypot(*lead_state.velocity_xy_mps) == pytest.approx(lead_speed_mps, abs=0.2)
            if "crossing-cyclist" in actor_ids:
                revealed = True
            if not revealed:
                assert "crossing-lead" in actor_ids
            elif "cleared_step" in step_info["scenario_parameters"]:
                cleared = True
                assert "crossing-cyclist" in actor_ids
            if terminated or truncated:
                assert truncated is False
                assert step_info["scenario_success"] is True
                completed = True
                break
        assert revealed is True
        assert cleared is True
        assert completed is True
        crossing_contact = environment._environment.scenario_ego_collided_with(
            "crossing-cyclist"
        )
        return tuple(positions), crossing_contact
    finally:
        environment.close()


@pytest.mark.integration
def test_real_occluded_crossing_reports_cyclist_collision_as_scenario_failure() -> None:
    environment = MultiAgentSpeedEnv(
        load_config("configs/base.yaml", "configs/scenarios/occluded_crossing.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        environment.reset(seed=42)
        manager = environment._environment.engine.scenario_actor_manager
        cyclist = manager.engine.get_objects(["crossing-cyclist"])["crossing-cyclist"]
        cyclist.set_position(environment._environment.vehicle.position)

        _observation, _reward, terminated, truncated, info = environment.step(3)

        assert terminated is True
        assert truncated is False
        assert info["crash_human"] is True
        assert info["scenario_failure"] is True
        frame = environment._frame
        assert frame is not None
        assert frame.privileged.collision_kind == "crossing_actor"
    finally:
        environment.close()


@pytest.mark.integration
def test_real_occluded_crossing_is_hidden_then_revealed_deterministic_and_cleans_up() -> None:
    first_positions, first_contact = run_occluded_crossing_prefix()
    second_positions, second_contact = run_occluded_crossing_prefix()

    assert first_positions == second_positions
    assert first_contact is False
    assert second_contact is False

    environment = MultiAgentSpeedEnv(
        load_config("configs/base.yaml", "configs/scenarios/occluded_crossing.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        environment.reset(seed=42)
        manager = environment._environment.engine.scenario_actor_manager
        previous_cyclist = manager.engine.get_objects(["crossing-cyclist"])["crossing-cyclist"]
        environment.reset(seed=43)

        assert previous_cyclist not in manager.engine.get_objects().values()
        assert manager.actor_ids() == ("crossing-cyclist", "static-occluder", "crossing-lead")
    finally:
        environment.close()
