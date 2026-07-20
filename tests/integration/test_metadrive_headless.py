import math
from dataclasses import asdict
from typing import Any

import pytest

from mad_driving.config.loader import load_config
from mad_driving.envs.multi_agent_speed_env import create_metadrive_env
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
        reset_result = env.reset(seed=config.seed)
        assert isinstance(reset_result, tuple)
        assert len(reset_result) == 2

        step_result = env.step(config.fixed_action)
        assert isinstance(step_result, tuple)
        assert len(step_result) == 5

        snapshot = SceneSnapshotBuilder().build(
            env,
            step_index=1,
            scenario_id=config.scenario_id,
            seed=config.seed,
            previous_action=0,
            previous_shield_intervention=False,
        )
        assert snapshot.step_index == 1
        assert snapshot.ego.speed_mps >= 0.0
        assert_finite_tree(asdict(snapshot))
    finally:
        env.close()
