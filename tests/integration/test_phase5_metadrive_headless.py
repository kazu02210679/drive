import json
import math
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from metadrive.component.static_object.traffic_object import TrafficObject
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.vehicle.base_vehicle import BaseVehicle

from mad_driving.agents.hazard import HazardAgent
from mad_driving.agents.kinematics import project_vector
from mad_driving.agents.nominal import NominalMotionAgent
from mad_driving.config.loader import load_config
from mad_driving.config.models import AppConfig
from mad_driving.envs import MultiAgentSpeedEnv
from mad_driving.interfaces import DecisionTrace

ALLOWED_SCENARIOS_BY_LEVEL = {
    0: {"nominal"},
    1: {"lead_brake"},
    2: {"lead_brake", "cut_in"},
    3: {"occluded_crossing"},
}


def fixed_level_config(level: int) -> AppConfig:
    payload = load_config("configs/base.yaml").model_dump(mode="python")
    payload["scenario_id"] = "phase5"
    payload["scenarios"]["selection"] = {
        0: "nominal",
        1: "lead_brake",
        2: "auto",
        3: "occluded_crossing",
    }[level]
    payload["scenarios"]["curriculum"] = {
        "mode": "fixed",
        "fixed_level": level,
    }
    return AppConfig.model_validate(payload)


def run_fixed_level_replay(level: int, episode_seed: int) -> bytes:
    config = fixed_level_config(level)
    environment = MultiAgentSpeedEnv(
        config,
        role="train",
        worker_index=0,
    )
    try:
        observation, reset_info = environment.reset(seed=episode_seed)
        assert observation.shape == (24,)
        assert np.isfinite(observation).all()
        manager = environment._environment.engine.scenario_actor_manager
        ego_lane = environment._environment.vehicle.navigation.current_lane
        trajectory: list[dict[str, object]] = []
        reached_episode_boundary = False
        for step_index in range(1, config.metadrive.horizon + 2):
            observation, reward, terminated, truncated, info = environment.step(3)
            assert observation.shape == (24,)
            assert np.isfinite(observation).all()
            assert math.isfinite(reward)
            actor_states: dict[str, object] = {}
            for actor_id in manager.actor_ids():
                state = manager.actor_state(actor_id)
                _, lateral_m = ego_lane.local_coordinates(state.position_xy_m)
                actor_states[actor_id] = {
                    "position_xy_m": list(state.position_xy_m),
                    "velocity_xy_mps": list(state.velocity_xy_mps),
                    "speed_mps": math.hypot(*state.velocity_xy_mps),
                    "ego_lane_lateral_m": lateral_m,
                }
            frame = environment._frame
            assert frame is not None
            trace = info["decision_trace"]
            assert isinstance(trace, DecisionTrace)
            trajectory.append(
                {
                    "step": step_index,
                    "observation": observation.tolist(),
                    "reward": reward,
                    "actors": actor_states,
                    "visible_actor_ids": sorted(
                        actor.actor_id for actor in frame.observation.visible_actors
                    ),
                    "scenario_parameters": info["scenario_parameters"],
                    "scenario_success": info["scenario_success"],
                    "scenario_failure": info["scenario_failure"],
                    "collision_occurred": info["collision_occurred"],
                    "terminated": terminated,
                    "truncated": truncated,
                    "trace_metadata": {
                        "environment_seed": trace.episode_rng_seed,
                        "scenario_selection_seed": trace.metadrive_scenario_index,
                        "scenario_parameter_seed": trace.scenario_parameter_seed,
                        "scenario_id": trace.scenario_id,
                        "difficulty_level": trace.difficulty_level,
                        "role": trace.role,
                        "worker_index": trace.worker_index,
                    },
                }
            )
            if terminated or truncated or info["scenario_success"] or info["scenario_failure"]:
                reached_episode_boundary = True
                break
        assert reached_episode_boundary is True
        return json.dumps(
            {
                "episode_seed": episode_seed,
                "scenario_id": reset_info["scenario_id"],
                "difficulty_level": reset_info["difficulty_level"],
                "scenario_parameters": reset_info["scenario_parameters"],
                "trajectory": trajectory,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    finally:
        environment.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("level", "episode_seed", "expected_scenario"),
    [
        (0, 42, "nominal"),
        (1, 42, "lead_brake"),
        (2, 43, "lead_brake"),
        (2, 42, "cut_in"),
        (3, 42, "occluded_crossing"),
    ],
)
def test_real_fixed_levels_select_only_allowed_scenarios_and_replay_byte_for_byte(
    level: int,
    episode_seed: int,
    expected_scenario: str,
) -> None:
    first = run_fixed_level_replay(level, episode_seed)
    second = run_fixed_level_replay(level, episode_seed)

    assert first == second
    payload = json.loads(first)
    assert payload["scenario_id"] == expected_scenario
    assert payload["scenario_id"] in ALLOWED_SCENARIOS_BY_LEVEL[level]
    assert payload["difficulty_level"] == level
    trajectory = payload["trajectory"]
    assert trajectory
    assert all(
        item["trace_metadata"]["scenario_id"] == payload["scenario_id"] for item in trajectory
    )
    assert all(item["trace_metadata"]["difficulty_level"] == level for item in trajectory)
    final = trajectory[-1]
    assert (
        final["terminated"]
        or final["truncated"]
        or final["scenario_success"]
        or final["scenario_failure"]
    )
    if expected_scenario == "nominal":
        assert all(not item["actors"] for item in trajectory)
        assert final["scenario_success"] is True
    elif expected_scenario == "lead_brake":
        speeds = [item["actors"]["lead-brake"]["speed_mps"] for item in trajectory]
        trigger_step = payload["scenario_parameters"]["trigger_step"]
        post_trigger_speeds = speeds[trigger_step - 1 :]
        assert len(post_trigger_speeds) >= 3
        assert sum(later < earlier for earlier, later in pairwise(post_trigger_speeds)) >= 2
        assert final["scenario_success"] is True
    elif expected_scenario == "cut_in":
        laterals = [item["actors"]["cut-in"]["ego_lane_lateral_m"] for item in trajectory]
        assert abs(laterals[0]) > 0.1
        assert min(abs(value) for value in laterals) <= 0.1
        assert final["scenario_success"] is True
    else:
        cyclist_speeds = [item["actors"]["crossing-cyclist"]["speed_mps"] for item in trajectory]
        cyclist_visible = ["crossing-cyclist" in item["visible_actor_ids"] for item in trajectory]
        assert cyclist_visible[0] is False
        assert any(cyclist_visible)
        assert max(cyclist_speeds) > 0.1
        assert all("crossing-lead" in item["actors"] for item in trajectory)
        assert final["scenario_success"] is True


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
        assert sum(later < earlier for earlier, later in pairwise(braking_speeds)) >= 2
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


@pytest.mark.integration
def test_real_lead_brake_frame_reports_braking_acceleration_to_nominal_agent() -> None:
    config = load_config("configs/base.yaml", "configs/scenarios/lead_brake.yaml")
    environment = MultiAgentSpeedEnv(config, role="train", worker_index=0)
    try:
        _observation, info = environment.reset(seed=42)
        trigger_step = info["scenario_parameters"]["trigger_step"]
        sampled_deceleration = info["scenario_parameters"]["deceleration_mps2"]
        assert isinstance(trigger_step, int)
        assert isinstance(sampled_deceleration, float)
        manager = environment._environment.engine.scenario_actor_manager
        simulator_lead = manager.engine.get_objects(["lead-brake"])["lead-brake"]
        simulator_lead.set_velocity(
            (math.cos(simulator_lead.heading_theta), math.sin(simulator_lead.heading_theta)),
            8.0,
        )

        for _ in range(trigger_step):
            _observation, _reward, terminated, truncated, _info = environment.step(0)
            assert not (terminated or truncated)

        frame = environment._frame
        assert frame is not None
        lead = next(
            actor for actor in frame.observation.visible_actors if actor.actor_id == "lead-brake"
        )
        longitudinal_acceleration, _ = project_vector(
            lead.acceleration_xy_mps2,
            frame.observation.ego.heading_rad,
        )
        assert math.isfinite(longitudinal_acceleration)
        assert longitudinal_acceleration == pytest.approx(-sampled_deceleration, abs=0.5)

        nominal = NominalMotionAgent(config.agents.nominal)
        braking_claim = next(
            claim
            for claim in nominal.analyze(frame.observation)
            if claim.target_actor_id == "lead-brake"
        )
        zero_acceleration_observation = replace(
            frame.observation,
            visible_actors=tuple(
                replace(actor, acceleration_xy_mps2=(0.0, 0.0))
                if actor.actor_id == "lead-brake"
                else actor
                for actor in frame.observation.visible_actors
            ),
        )
        zero_acceleration_claim = next(
            claim
            for claim in nominal.analyze(zero_acceleration_observation)
            if claim.target_actor_id == "lead-brake"
        )
        assert braking_claim.probability is not None
        assert zero_acceleration_claim.probability is not None
        assert braking_claim.probability > zero_acceleration_claim.probability
        assert (
            braking_claim.recommended_max_speed_mps
            < zero_acceleration_claim.recommended_max_speed_mps
        )
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


@pytest.mark.integration
def test_real_cut_in_frame_relocalizes_merged_actor_to_the_ego_lane() -> None:
    config = load_config("configs/base.yaml", "configs/scenarios/cut_in.yaml")
    environment = MultiAgentSpeedEnv(config, role="train", worker_index=0)
    try:
        _observation, info = environment.reset(seed=42)
        parameters = info["scenario_parameters"]
        trigger_step = parameters["trigger_step"]
        merge_steps = parameters["merge_steps"]
        assert isinstance(trigger_step, int)
        assert isinstance(merge_steps, int)

        for _ in range(trigger_step + merge_steps):
            _observation, _reward, terminated, truncated, _info = environment.step(3)
            assert not (terminated or truncated)

        frame = environment._frame
        assert frame is not None
        cut_in = next(
            actor for actor in frame.observation.visible_actors if actor.actor_id == "cut-in"
        )
        assert abs(cut_in.relative_lateral_m) < 0.1
        assert cut_in.same_lane is True

        hazard_claims = HazardAgent(config.agents.hazard).analyze(frame.observation)
        assert any(
            claim.target_actor_id == "cut-in" and claim.event_type == "hazard_lead_braking"
            for claim in hazard_claims
        )
        nominal_claims = NominalMotionAgent(config.agents.nominal).analyze(frame.observation)
        assert any(
            claim.target_actor_id == "cut-in" and claim.event_type == "nominal_lead"
            for claim in nominal_claims
        )
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
        crossing_contact = environment._environment.scenario_ego_collided_with("crossing-cyclist")
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
def test_real_occluder_contact_is_attributed_as_a_static_object_collision() -> None:
    environment = MultiAgentSpeedEnv(
        load_config("configs/base.yaml", "configs/scenarios/occluded_crossing.yaml"),
        role="train",
        worker_index=0,
    )
    try:
        environment.reset(seed=42)
        manager = environment._environment.engine.scenario_actor_manager
        occluder = manager.engine.get_objects(["static-occluder"])["static-occluder"]
        assert occluder.id == "static-occluder"
        assert environment._environment.scenario_ego_collided_with("static-occluder") is False

        occluder.set_position(environment._environment.vehicle.position)
        _observation, _reward, terminated, truncated, info = environment.step(3)

        assert terminated is True
        assert truncated is False
        assert info["crash_object"] is True
        assert info["crash_vehicle"] is False
        frame = environment._frame
        assert frame is not None
        assert frame.privileged.collision_kind == "object"
        assert occluder.LENGTH == pytest.approx(5.0)
        assert occluder.WIDTH == pytest.approx(2.0)
        privileged_occluder = next(
            actor for actor in frame.privileged.all_actors if actor.actor_id == "static-occluder"
        )
        assert privileged_occluder.actor_type == "obstacle"
        assert isinstance(occluder, TrafficObject)
        assert not isinstance(occluder, BaseVehicle)
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
