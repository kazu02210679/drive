import math

import pytest

from mad_driving.config.loader import load_config
from mad_driving.control import DrivingAction
from mad_driving.envs.control_metadrive_env import create_control_metadrive_env


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
