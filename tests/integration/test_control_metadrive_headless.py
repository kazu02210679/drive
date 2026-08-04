import math
from dataclasses import asdict
from typing import Any

import pytest

from mad_driving.agents.suite import AgentAnalysisResult
from mad_driving.cli.control_smoke import run_control_smoke
from mad_driving.config.loader import load_config
from mad_driving.config.models import AppConfig
from mad_driving.control import DrivingAction, LaneKeepingLongitudinalPolicy
from mad_driving.envs.control_metadrive_env import create_control_metadrive_env
from mad_driving.envs.multi_agent_speed_env import MultiAgentSpeedEnv
from mad_driving.interfaces import CriticReview, DecisionTrace, RiskClaim, SceneObservation


def assert_finite_tree(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            assert_finite_tree(child)
    elif isinstance(value, tuple | list):
        for child in value:
            assert_finite_tree(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


class EmergencyOnSecondAnalysisSuite:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, observation: SceneObservation) -> AgentAnalysisResult:
        self.calls += 1
        hard_stop = self.calls >= 2
        claims = tuple(
            RiskClaim(
                claim_id=f"{agent_id}:{observation.step_index}:{self.calls}",
                agent_id=agent_id,
                event_type="integration_emergency",
                target_actor_id=None,
                probability=1.0 if hard_stop and agent_id == "hazard" else 0.0,
                confidence=1.0,
                severity=1.0 if hard_stop and agent_id == "hazard" else 0.0,
                time_horizon_s=1.0,
                min_ttc_s=0.5 if hard_stop and agent_id == "hazard" else None,
                stopping_margin_m=-1.0 if hard_stop and agent_id == "hazard" else None,
                recommended_max_speed_mps=(0.0 if hard_stop and agent_id == "hazard" else 100.0),
                hard_stop_required=hard_stop and agent_id == "hazard",
                evidence=("real_metadrive_policy_path",),
                assumptions=(),
                valid_until_step=observation.step_index + 1,
            )
            for agent_id in ("nominal", "hazard", "rule")
        )
        return AgentAnalysisResult(
            claims=claims,
            failed_agent_ids=(),
            errors=(),
            review=CriticReview(
                conflict_score=0.0,
                unresolved_conflict=False,
                max_severity=max(claim.severity for claim in claims),
                supported_agent_ids=("nominal", "hazard", "rule"),
                challenged_claim_ids=(),
                reasons=(),
            ),
            expected_agent_ids=("nominal", "hazard", "rule"),
        )


def shield_mode_config(mode: str) -> AppConfig:
    payload = load_config("configs/base.yaml").model_dump(mode="python")
    payload["shield"]["mode"] = mode
    return AppConfig.model_validate(payload)


def controlled_policy(environment: Any) -> LaneKeepingLongitudinalPolicy:
    policies = tuple(environment.engine._object_policies.values())
    matching = tuple(
        policy for policy in policies if isinstance(policy, LaneKeepingLongitudinalPolicy)
    )
    assert len(matching) == 1
    return matching[0]


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


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mode", "expected_executed", "expected_intervened"),
    [
        ("enforce", DrivingAction.STOP, True),
        ("monitor", DrivingAction.KEEP, False),
    ],
)
def test_shield_emergency_reaches_real_policy_only_when_enforced(
    mode: str,
    expected_executed: DrivingAction,
    expected_intervened: bool,
) -> None:
    config = shield_mode_config(mode)
    created = []

    def factory(options: dict[str, object], control: Any):
        environment = create_control_metadrive_env(options, control)
        created.append(environment)
        return environment

    env = MultiAgentSpeedEnv(
        config,
        role="train",
        worker_index=0,
        env_factory=factory,
        suite_factory=lambda _: EmergencyOnSecondAnalysisSuite(),
    )
    try:
        env.reset(seed=42)
        _, _, terminated, truncated, first_info = env.step(int(DrivingAction.SLOW))
        assert not (terminated or truncated)
        assert first_info["executed_action"] == int(DrivingAction.SLOW)
        policy = controlled_policy(created[0])
        assert policy._speed_pid.previous_error is not None

        _, _, _, _, emergency_info = env.step(int(DrivingAction.KEEP))
        assert emergency_info["requested_action"] == int(DrivingAction.KEEP)
        assert emergency_info["executed_action"] == int(expected_executed)
        assert emergency_info["shield_intervened"] is expected_intervened
        assert "hard_stop_required" in emergency_info["shield_reasons"]
        assert policy.action_info["requested_action"] == int(expected_executed)

        if mode == "enforce":
            assert policy.action_info["target_speed_mps"] == 0.0
            assert policy.action_info["throttle_brake"] == -1.0
            assert policy._speed_pid.previous_error is None
            assert (
                policy._control_config.speed.emergency_deceleration_mps2
                == config.control.speed.emergency_deceleration_mps2
            )
        else:
            assert policy._speed_pid.previous_error is not None
    finally:
        env.close()
