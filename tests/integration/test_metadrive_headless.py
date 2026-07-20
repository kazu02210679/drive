import math
from dataclasses import asdict
from typing import Any

import pytest

from mad_driving.agents import AgentSuite
from mad_driving.config.loader import load_config
from mad_driving.envs.multi_agent_speed_env import create_metadrive_env
from mad_driving.scenarios import (
    EpisodeSeeds,
    NoOpScenarioRuntime,
    ScenarioStepResult,
)
from mad_driving.world_model import SceneSnapshotBuilder


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
def test_real_metadrive_headless_step_builds_finite_snapshot() -> None:
    config = load_config("configs/base.yaml")
    env = create_metadrive_env(config.metadrive_dict())

    try:
        seeds = EpisodeSeeds(config.seed, config.seed, config.seed)
        runtime = NoOpScenarioRuntime(config.scenario_id)
        state = runtime.reset(env, seeds=seeds)
        reset_result = env.reset(seed=seeds.metadrive_scenario_index)
        assert isinstance(reset_result, tuple)
        assert len(reset_result) == 2

        runtime.after_simulator_reset(env, state)
        runtime.before_step(env, state, step_index=1)
        step_result = env.step(config.fixed_action)
        assert isinstance(step_result, tuple)
        assert len(step_result) == 5

        raw_info = step_result[4]
        scenario_result = runtime.after_step(
            env,
            state,
            step_index=1,
            raw_info=raw_info,
        )
        assert scenario_result == ScenarioStepResult(False, False)
        frame = SceneSnapshotBuilder().build(
            env,
            step_index=1,
            seeds=seeds,
            context=runtime.observation_context(state),
            scenario_result=scenario_result,
            raw_info=raw_info,
            previous_executed_action=0,
            previous_shield_intervention=False,
        )
        snapshot = frame.observation
        assert snapshot.step_index == 1
        assert snapshot.ego.speed_mps >= 0.0
        analysis = AgentSuite.from_config(config.agents).analyze(snapshot)
        assert tuple(claim.agent_id for claim in analysis.claims) == (
            "nominal",
            "hazard",
            "rule",
        )
        assert_finite_tree(asdict(snapshot))
        assert_finite_tree([asdict(claim) for claim in analysis.claims])
        assert_finite_tree(asdict(analysis.review))
    finally:
        env.close()
