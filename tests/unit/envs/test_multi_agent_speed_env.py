from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray

from mad_driving.config.models import AppConfig, ControlConfig
from mad_driving.control import DrivingAction
from mad_driving.envs import MultiAgentSpeedEnv
from mad_driving.envs.reward import RewardContext, RewardResult
from mad_driving.interfaces import (
    CriticReview,
    DecisionTrace,
    RiskClaim,
    SceneSnapshot,
    ShieldResult,
)
from mad_driving.safety import SafetyShield
from tests.unit.agents.factories import make_claim, make_snapshot


class FakeLane:
    speed_limit = 54.0
    index = ("A", "B", 0)

    @staticmethod
    def local_coordinates(position: tuple[float, float]) -> tuple[float, float]:
        return position


class FakeNavigation:
    def __init__(self) -> None:
        self.current_lane = FakeLane()
        self.route_completion = 0.25


class FakeVehicle:
    def __init__(self) -> None:
        self.name = "ego"
        self.position = (0.0, 0.0)
        self.velocity = (10.0, 0.0)
        self.last_velocity = (10.0, 0.0)
        self.heading_theta = 0.0
        self.navigation = FakeNavigation()
        self.lane_index = FakeLane.index
        self.max_speed_m_s = 20.0
        self.speed = 10.0
        self.on_lane = True
        self.crash_vehicle = False
        self.crash_human = False
        self.crash_object = False
        self.crash_sidewalk = False
        self.crash_building = False


class FakeEngine:
    def __init__(self, vehicle: FakeVehicle) -> None:
        self._vehicle = vehicle

    def get_objects(self) -> dict[str, FakeVehicle]:
        return {"ego": self._vehicle}


class FakeSimulator:
    def __init__(
        self,
        *,
        step_results: Sequence[tuple[bool, bool, dict[str, Any]]] = (),
        fail_on_reset_calls: Sequence[int] = (),
        fail_on_step: bool = False,
        fail_on_close: bool = False,
    ) -> None:
        self.vehicle = FakeVehicle()
        self.agent = self.vehicle
        self.engine = FakeEngine(self.vehicle)
        self.action_space = gym.spaces.Discrete(4)
        self.config: dict[str, Any] = {
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
        }
        self.step_results = tuple(step_results)
        self.fail_on_reset_calls = set(fail_on_reset_calls)
        self.fail_on_step = fail_on_step
        self.fail_on_close = fail_on_close
        self.reset_seeds: list[int | None] = []
        self.actions: list[int] = []
        self.close_calls = 0
        self.reset_info: dict[str, object] = {"simulator_reset": True}

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, object], dict[str, object]]:
        self.reset_seeds.append(seed)
        if len(self.reset_seeds) in self.fail_on_reset_calls:
            raise RuntimeError("simulator reset failed")
        self.actions.clear()
        self.vehicle.position = (0.0, 0.0)
        self.vehicle.navigation.route_completion = 0.25
        self.vehicle.crash_vehicle = False
        self.vehicle.crash_human = False
        return {}, self.reset_info

    def step(self, action: int) -> tuple[dict[str, object], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        if self.fail_on_step:
            raise RuntimeError("simulator step failed")
        index = len(self.actions) - 1
        terminated, truncated, info = (
            self.step_results[index] if index < len(self.step_results) else (False, False, {})
        )
        self.vehicle.position = (float(len(self.actions)), 0.0)
        self.vehicle.navigation.route_completion += 0.1
        self.vehicle.crash_vehicle = bool(info.get("crash_vehicle", False))
        self.vehicle.crash_human = bool(info.get("crash_human", False))
        return {}, 999.0, terminated, truncated, info

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_on_close:
            raise RuntimeError("simulator close failed")


class RecordingEnvironmentFactory:
    def __init__(
        self,
        simulators: Sequence[FakeSimulator] = (),
        *,
        fail_on_calls: Sequence[int] = (),
    ) -> None:
        self._simulators = list(simulators)
        self._fail_on_calls = set(fail_on_calls)
        self.created: list[FakeSimulator] = []
        self.calls: list[tuple[dict[str, object], ControlConfig]] = []

    def __call__(
        self,
        options: dict[str, object],
        control_config: ControlConfig,
    ) -> FakeSimulator:
        self.calls.append((options, control_config))
        if len(self.calls) in self._fail_on_calls:
            raise RuntimeError("environment creation failed")
        simulator = self._simulators.pop(0) if self._simulators else FakeSimulator()
        self.created.append(simulator)
        return simulator


class ConfigurableFactory:
    def __init__(
        self,
        name: str,
        value: object,
        *,
        fail_on_calls: Sequence[int] = (),
    ) -> None:
        self.name = name
        self.value = value
        self.fail_on_calls = set(fail_on_calls)
        self.calls = 0

    def __call__(self, *args: object) -> Any:
        del args
        self.calls += 1
        if self.calls in self.fail_on_calls:
            raise RuntimeError(f"{self.name} factory failed")
        return self.value


def neutral_review(*, reason: str | None = None) -> CriticReview:
    return CriticReview(
        conflict_score=0.0 if reason is None else 1.0,
        unresolved_conflict=reason is not None,
        max_severity=0.0 if reason is None else 1.0,
        supported_agent_ids=(),
        challenged_claim_ids=(),
        reasons=() if reason is None else (reason,),
    )


def complete_claims(*, severity: float = 0.0) -> tuple[RiskClaim, ...]:
    return (
        make_claim("nominal", severity=severity),
        make_claim("hazard", severity=severity),
        make_claim("rule", severity=severity),
    )


class SequenceSuite:
    def __init__(
        self,
        results: Sequence[tuple[tuple[RiskClaim, ...], CriticReview]] = (),
    ) -> None:
        self.results = tuple(results) or ((complete_claims(), neutral_review()),)
        self.snapshots: list[SceneSnapshot] = []

    def analyze(self, snapshot: SceneSnapshot) -> tuple[tuple[RiskClaim, ...], CriticReview]:
        self.snapshots.append(snapshot)
        index = min(len(self.snapshots) - 1, len(self.results) - 1)
        return self.results[index]


class FailingSuite:
    def analyze(self, snapshot: SceneSnapshot) -> tuple[tuple[RiskClaim, ...], CriticReview]:
        del snapshot
        raise RuntimeError("agent analysis failed")


class RecordingSnapshotBuilder:
    def __init__(self, *, failing_calls: Sequence[int] = ()) -> None:
        self.failing_calls = set(failing_calls)
        self.calls: list[dict[str, Any]] = []

    def build(
        self,
        env: FakeSimulator,
        *,
        step_index: int,
        scenario_id: str,
        seed: int,
        previous_action: int,
        previous_shield_intervention: bool,
    ) -> SceneSnapshot:
        self.calls.append(
            {
                "env": env,
                "step_index": step_index,
                "scenario_id": scenario_id,
                "seed": seed,
                "previous_action": previous_action,
                "previous_shield_intervention": previous_shield_intervention,
            }
        )
        if len(self.calls) in self.failing_calls:
            raise RuntimeError("initial snapshot failed")
        return make_snapshot(
            step_index=step_index,
            scenario_id=scenario_id,
            seed=seed,
            previous_action=previous_action,
            previous_shield_intervention=previous_shield_intervention,
            collision_occurred=env.vehicle.crash_vehicle or env.vehicle.crash_human,
            ego_speed_mps=env.vehicle.speed,
            speed_limit_mps=FakeLane.speed_limit / 3.6,
        )


class RecordingObservationBuilder:
    def __init__(self, *, failing_calls: Sequence[int] = ()) -> None:
        self.failing_calls = set(failing_calls)
        self.calls: list[tuple[SceneSnapshot, tuple[RiskClaim, ...], CriticReview]] = []

    def build(
        self,
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> NDArray[np.float32]:
        self.calls.append((snapshot, tuple(claims), review))
        if len(self.calls) in self.failing_calls:
            raise ValueError("observation build failed")
        value = np.float32(min(snapshot.step_index / 10.0, 1.0))
        return np.full((24,), value, dtype=np.float32)


class RecordingRewardCalculator:
    def __init__(
        self,
        components: dict[str, float] | None = None,
        *,
        failing_reset_calls: Sequence[int] = (),
        failing_calculate_calls: Sequence[int] = (),
        invalid_trace_calls: Sequence[int] = (),
    ) -> None:
        self.components = components or {"progress_reward": 1.25, "safety_penalty": -0.25}
        self.failing_reset_calls = set(failing_reset_calls)
        self.failing_calculate_calls = set(failing_calculate_calls)
        self.invalid_trace_calls = set(invalid_trace_calls)
        self.contexts: list[RewardContext] = []
        self.reset_calls = 0
        self.calculate_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        if self.reset_calls in self.failing_reset_calls:
            raise RuntimeError("reward reset failed")

    def calculate(self, context: RewardContext) -> RewardResult:
        self.calculate_calls += 1
        self.contexts.append(context)
        if self.calculate_calls in self.failing_calculate_calls:
            raise RuntimeError("reward calculate failed")
        if self.calculate_calls in self.invalid_trace_calls:
            return InvalidTraceRewardResult()  # type: ignore[return-value]
        return RewardResult(total=sum(self.components.values()), components=self.components)


class InvalidTraceRewardResult:
    def __init__(self) -> None:
        self.total = 0.0
        self.components = {"trace_failure": float("nan")}


class RecordingShield:
    def __init__(self, executed_action: DrivingAction = DrivingAction.STOP) -> None:
        self.executed_action = executed_action
        self.calls: list[tuple[DrivingAction | int, SceneSnapshot, tuple[RiskClaim, ...]]] = []

    def filter(
        self,
        requested_action: DrivingAction | int,
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
    ) -> ShieldResult:
        self.calls.append((requested_action, snapshot, tuple(claims)))
        requested = DrivingAction(requested_action)
        required = max(requested, self.executed_action)
        executed = max(requested, self.executed_action)
        return ShieldResult(
            requested_action=requested,
            required_action=required,
            executed_action=executed,
            intervention_required=required > requested,
            intervened=executed != requested,
            reasons=("fake_stop",) if executed != requested else (),
        )


def make_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "seed": 42,
            "scenario_id": "unit_multi_agent_speed_env",
            "decision_steps": 8,
            "fixed_action": [0.0, 0.25],
            "metadrive": {"use_render": False},
        }
    )


@dataclass
class EnvHarness:
    env: MultiAgentSpeedEnv
    env_factory: RecordingEnvironmentFactory
    suite: SequenceSuite | FailingSuite
    shield: RecordingShield | SafetyShield
    snapshot_builder: RecordingSnapshotBuilder
    reward: RecordingRewardCalculator
    observation: RecordingObservationBuilder


def make_env(
    *,
    simulators: Sequence[FakeSimulator] = (),
    env_factory: RecordingEnvironmentFactory | None = None,
    suite: SequenceSuite | FailingSuite | None = None,
    shield: RecordingShield | SafetyShield | None = None,
    snapshot_builder: RecordingSnapshotBuilder | None = None,
    reward: RecordingRewardCalculator | None = None,
    observation: RecordingObservationBuilder | None = None,
    suite_factory: Callable[..., Any] | None = None,
    shield_factory: Callable[..., Any] | None = None,
    builder_factory: Callable[..., Any] | None = None,
    reward_factory: Callable[..., Any] | None = None,
    observation_factory: Callable[..., Any] | None = None,
) -> EnvHarness:
    selected_env_factory = env_factory or RecordingEnvironmentFactory(simulators)
    selected_suite = suite or SequenceSuite()
    selected_shield = shield or RecordingShield()
    selected_snapshot_builder = snapshot_builder or RecordingSnapshotBuilder()
    selected_reward = reward or RecordingRewardCalculator()
    selected_observation = observation or RecordingObservationBuilder()
    env = MultiAgentSpeedEnv(
        make_config(),
        env_factory=selected_env_factory,
        suite_factory=suite_factory or (lambda config: selected_suite),
        shield_factory=shield_factory or (lambda config: selected_shield),
        builder_factory=builder_factory or (lambda: selected_snapshot_builder),
        reward_factory=reward_factory or (lambda config: selected_reward),
        observation_factory=observation_factory or (lambda config: selected_observation),
    )
    return EnvHarness(
        env=env,
        env_factory=selected_env_factory,
        suite=selected_suite,
        shield=selected_shield,
        snapshot_builder=selected_snapshot_builder,
        reward=selected_reward,
        observation=selected_observation,
    )


def test_env_exposes_fixed_spaces_and_seeded_reset() -> None:
    harness = make_env()
    try:
        first, info = harness.env.reset(seed=123)
        second, second_info = harness.env.reset(seed=123)

        assert harness.env.action_space == gym.spaces.Discrete(4)
        assert harness.env.observation_space.shape == (24,)
        assert harness.env.observation_space.dtype == np.float32
        assert harness.env.observation_space.contains(first)
        np.testing.assert_array_equal(first, second)
        assert info["seed"] == 123
        assert info["simulator_reset"] is True
        assert second_info["seed"] == 123
        assert harness.env_factory.created[0].reset_seeds == [123, 123]
        assert harness.reward.reset_calls == 2
    finally:
        harness.env.close()


def test_reset_uses_config_seed_when_seed_is_omitted() -> None:
    harness = make_env()
    try:
        _, info = harness.env.reset()
        assert info["seed"] == 42
        assert harness.env_factory.created[0].reset_seeds == [42]
    finally:
        harness.env.close()


def test_simulator_creation_is_lazy_and_first_failure_can_be_retried() -> None:
    replacement = FakeSimulator()
    env_factory = RecordingEnvironmentFactory((replacement,), fail_on_calls=(1,))
    harness = make_env(env_factory=env_factory)
    try:
        assert env_factory.calls == []

        with pytest.raises(RuntimeError, match="environment creation failed"):
            harness.env.reset(seed=5)
        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)

        observation, info = harness.env.reset(seed=6)
        assert env_factory.created == [replacement]
        assert replacement.reset_seeds == [6]
        assert harness.env.observation_space.contains(observation)
        assert info["seed"] == 6
    finally:
        harness.env.close()


@pytest.mark.parametrize(
    "failing_factory_name",
    ["suite", "shield", "builder", "reward", "observation"],
)
def test_second_reset_dependency_factory_failure_is_transactional_and_retryable(
    failing_factory_name: str,
) -> None:
    first = FakeSimulator()
    replacement = FakeSimulator()
    suite = SequenceSuite()
    shield = RecordingShield()
    snapshot_builder = RecordingSnapshotBuilder()
    reward = RecordingRewardCalculator()
    observation = RecordingObservationBuilder()
    factories = {
        "suite": ConfigurableFactory("suite", suite),
        "shield": ConfigurableFactory("shield", shield),
        "builder": ConfigurableFactory("builder", snapshot_builder),
        "reward": ConfigurableFactory("reward", reward),
        "observation": ConfigurableFactory("observation", observation),
    }
    factories[failing_factory_name].fail_on_calls.add(2)
    harness = make_env(
        simulators=(first, replacement),
        suite=suite,
        shield=shield,
        snapshot_builder=snapshot_builder,
        reward=reward,
        observation=observation,
        suite_factory=factories["suite"],
        shield_factory=factories["shield"],
        builder_factory=factories["builder"],
        reward_factory=factories["reward"],
        observation_factory=factories["observation"],
    )
    try:
        harness.env.reset(seed=10)

        with pytest.raises(RuntimeError, match=f"{failing_factory_name} factory failed"):
            harness.env.reset(seed=11)

        assert first.close_calls == 1
        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)

        observation_value, info = harness.env.reset(seed=12)
        assert harness.env_factory.created == [first, replacement]
        assert replacement.reset_seeds == [12]
        assert harness.env.observation_space.contains(observation_value)
        assert info["seed"] == 12
    finally:
        harness.env.close()


def test_second_reset_reward_reset_failure_clears_stale_episode() -> None:
    first = FakeSimulator()
    replacement = FakeSimulator()
    reward = RecordingRewardCalculator(failing_reset_calls=(2,))
    harness = make_env(simulators=(first, replacement), reward=reward)
    try:
        harness.env.reset(seed=20)

        with pytest.raises(RuntimeError, match="reward reset failed"):
            harness.env.reset(seed=21)

        assert first.close_calls == 1
        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)

        observation, _ = harness.env.reset(seed=22)
        assert harness.env.observation_space.contains(observation)
        assert replacement.reset_seeds == [22]
    finally:
        harness.env.close()


@pytest.mark.parametrize("failure_kind", ["simulator_reset", "snapshot"])
def test_initial_reset_pipeline_failure_closes_and_can_retry(failure_kind: str) -> None:
    failed = FakeSimulator(fail_on_reset_calls=(1,) if failure_kind == "simulator_reset" else ())
    replacement = FakeSimulator()
    snapshot_builder = RecordingSnapshotBuilder(
        failing_calls=(1,) if failure_kind == "snapshot" else ()
    )
    harness = make_env(
        simulators=(failed, replacement),
        snapshot_builder=snapshot_builder,
    )
    try:
        with pytest.raises(RuntimeError, match="simulator reset failed|initial snapshot failed"):
            harness.env.reset(seed=30)

        assert failed.close_calls == 1
        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)

        observation, info = harness.env.reset(seed=31)
        assert replacement.reset_seeds == [31]
        assert harness.env.observation_space.contains(observation)
        assert info["seed"] == 31
    finally:
        harness.env.close()


def test_initial_observation_failure_closes_and_can_retry() -> None:
    failed = FakeSimulator()
    replacement = FakeSimulator()
    observation_builder = RecordingObservationBuilder(failing_calls=(1,))
    harness = make_env(
        simulators=(failed, replacement),
        observation=observation_builder,
    )
    try:
        with pytest.raises(ValueError, match="observation build failed"):
            harness.env.reset(seed=40)

        assert failed.close_calls == 1
        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)

        observation, info = harness.env.reset(seed=41)
        assert replacement.reset_seeds == [41]
        assert harness.env.observation_space.contains(observation)
        assert info["seed"] == 41
    finally:
        harness.env.close()


def test_step_runs_shielded_pipeline_and_builds_trace_from_both_sides() -> None:
    pre_claims = complete_claims(severity=0.1)
    post_claims = complete_claims(severity=0.9)
    pre_review = neutral_review()
    post_review = neutral_review(reason="post_step_review")
    suite = SequenceSuite(((pre_claims, pre_review), (post_claims, post_review)))
    shield = RecordingShield(DrivingAction.STOP)
    harness = make_env(suite=suite, shield=shield)
    try:
        harness.env.reset(seed=7)
        observation, reward, terminated, truncated, info = harness.env.step(DrivingAction.KEEP)

        simulator = harness.env_factory.created[0]
        assert shield.calls[0][0] == DrivingAction.KEEP
        assert shield.calls[0][2] == pre_claims
        assert simulator.actions == [int(DrivingAction.STOP)]
        assert harness.observation.calls[-1][1] == post_claims
        assert harness.observation.calls[-1][2] == post_review
        assert harness.reward.contexts[-1].post_step_claims == post_claims
        assert harness.reward.contexts[-1].previous_snapshot == suite.snapshots[0]
        assert harness.reward.contexts[-1].next_snapshot == suite.snapshots[1]
        assert info["requested_action"] == int(DrivingAction.KEEP)
        assert info["executed_action"] == int(DrivingAction.STOP)
        assert info["shield_intervened"] is True
        assert info["shield_reasons"] == ("fake_stop",)
        assert info["target_speed_mps"] == 0.0
        assert info["decision_trace"].claims == pre_claims
        assert info["decision_trace"].review == pre_review
        assert info["decision_trace"].reward_components == info["reward_components"]
        assert reward == pytest.approx(sum(info["reward_components"].values()))
        assert harness.env.observation_space.contains(observation)
        assert terminated is False
        assert truncated is False
    finally:
        harness.env.close()


def test_reset_and_step_return_exact_gymnasium_runtime_types() -> None:
    harness = make_env(shield=RecordingShield(DrivingAction.KEEP))
    try:
        reset_result = harness.env.reset(seed=70)
        assert type(reset_result) is tuple
        initial_observation, reset_info = reset_result
        assert type(initial_observation) is np.ndarray
        assert initial_observation.dtype == np.float32
        assert type(reset_info) is dict

        step_result = harness.env.step(0)
        assert type(step_result) is tuple
        observation, reward, terminated, truncated, info = step_result
        assert type(observation) is np.ndarray
        assert observation.dtype == np.float32
        assert type(reward) is float
        assert type(terminated) is bool
        assert type(truncated) is bool
        assert type(info) is dict
    finally:
        harness.env.close()


def test_returned_info_and_reward_components_do_not_alias_internal_values() -> None:
    raw_step_info: dict[str, Any] = {"simulator_value": "original"}
    simulator = FakeSimulator(step_results=((False, False, raw_step_info),))
    harness = make_env(simulators=(simulator,))
    try:
        _, reset_info = harness.env.reset()
        reset_info["simulator_reset"] = False
        assert simulator.reset_info == {"simulator_reset": True}

        _, _, _, _, info = harness.env.step(DrivingAction.KEEP)
        info["simulator_value"] = "changed"
        assert raw_step_info == {"simulator_value": "original"}

        components = info["reward_components"]
        trace = info["decision_trace"]
        assert isinstance(components, dict)
        assert isinstance(trace, DecisionTrace)
        components["progress_reward"] = 99.0
        assert trace.reward_components["progress_reward"] == 1.25
        assert harness.reward.components["progress_reward"] == 1.25

        trace.reward_components["progress_reward"] = 77.0
        assert components["progress_reward"] == 99.0
        assert harness.reward.components["progress_reward"] == 1.25
    finally:
        harness.env.close()


def test_step_before_reset_is_rejected() -> None:
    harness = make_env()
    try:
        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)
    finally:
        harness.env.close()


def test_numpy_integer_action_from_discrete_space_is_accepted() -> None:
    harness = make_env(shield=RecordingShield(DrivingAction.KEEP))
    try:
        harness.env.reset()
        sampled_action = harness.env.action_space.sample()
        assert isinstance(sampled_action, np.integer)
        harness.env.step(sampled_action)
        assert harness.env_factory.created[0].actions == [int(sampled_action)]
    finally:
        harness.env.close()


def test_python_int_action_is_accepted_without_coercing_its_value() -> None:
    harness = make_env(shield=RecordingShield(DrivingAction.KEEP))
    try:
        harness.env.reset()
        action = 1
        assert type(action) is int

        harness.env.step(action)

        assert harness.env_factory.created[0].actions == [1]
    finally:
        harness.env.close()


@pytest.mark.parametrize("action", [True, False, 1.0, np.float32(1.0), -1, 4])
def test_invalid_actions_are_rejected_without_clipping(action: object) -> None:
    harness = make_env()
    try:
        harness.env.reset()
        with pytest.raises(ValueError, match="action"):
            harness.env.step(action)  # type: ignore[arg-type]
        assert harness.env_factory.created[0].actions == []
    finally:
        harness.env.close()


def test_close_is_idempotent_and_swallows_simulator_close_error() -> None:
    simulator = FakeSimulator(fail_on_close=True)
    harness = make_env(simulators=(simulator,))

    harness.env.reset()
    harness.env.close()
    harness.env.close()

    assert simulator.close_calls == 1


def test_reset_after_close_recreates_the_simulator() -> None:
    first = FakeSimulator()
    second = FakeSimulator()
    harness = make_env(simulators=(first, second))
    try:
        harness.env.reset(seed=8)
        harness.env.close()
        observation, info = harness.env.reset(seed=9)

        assert first.close_calls == 1
        assert harness.env_factory.created == [first, second]
        assert second.reset_seeds == [9]
        assert harness.env.observation_space.contains(observation)
        assert info["seed"] == 9
    finally:
        harness.env.close()


def test_reset_after_close_error_recreates_the_simulator() -> None:
    first = FakeSimulator(fail_on_close=True)
    replacement = FakeSimulator()
    harness = make_env(simulators=(first, replacement))
    try:
        harness.env.reset(seed=50)
        harness.env.close()

        observation, info = harness.env.reset(seed=51)

        assert first.close_calls == 1
        assert harness.env_factory.created == [first, replacement]
        assert replacement.reset_seeds == [51]
        assert harness.env.observation_space.contains(observation)
        assert info["seed"] == 51
    finally:
        harness.env.close()


@pytest.mark.parametrize(
    ("raw_terminated", "raw_truncated"),
    [(True, False), (False, True), (True, True)],
)
def test_step_preserves_raw_termination_flags(
    raw_terminated: bool,
    raw_truncated: bool,
) -> None:
    simulator = FakeSimulator(step_results=((raw_terminated, raw_truncated, {}),))
    harness = make_env(simulators=(simulator,))
    try:
        harness.env.reset()
        _, _, terminated, truncated, _ = harness.env.step(DrivingAction.KEEP)
        assert terminated is raw_terminated
        assert truncated is raw_truncated
    finally:
        harness.env.close()


@pytest.mark.parametrize(
    ("raw_terminated", "raw_truncated"),
    [(True, False), (False, True)],
)
def test_terminal_step_rejects_further_steps_and_reset_starts_clean(
    raw_terminated: bool,
    raw_truncated: bool,
) -> None:
    simulator = FakeSimulator(step_results=((raw_terminated, raw_truncated, {}),))
    harness = make_env(simulators=(simulator,))
    try:
        harness.env.reset(seed=80)
        _, _, terminated, truncated, _ = harness.env.step(DrivingAction.KEEP)
        assert terminated is raw_terminated
        assert truncated is raw_truncated

        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)

        observation, info = harness.env.reset(seed=81)
        assert simulator.reset_seeds == [80, 81]
        assert simulator.actions == []
        assert harness.snapshot_builder.calls[-1]["step_index"] == 0
        assert harness.snapshot_builder.calls[-1]["previous_action"] == int(DrivingAction.KEEP)
        assert harness.env.observation_space.contains(observation)
        assert info["seed"] == 81

        harness.env.step(DrivingAction.KEEP)
    finally:
        harness.env.close()


@pytest.mark.parametrize(
    ("raw_info", "arrived", "collision_kind"),
    [
        ({"arrive_dest": True}, True, None),
        ({"crash_vehicle": True}, False, "vehicle"),
        ({"crash_human": True}, False, "crossing_actor"),
        ({"crash_vehicle": True, "crash_human": True}, False, "crossing_actor"),
    ],
)
def test_reward_context_maps_arrival_and_collision_info(
    raw_info: dict[str, bool],
    arrived: bool,
    collision_kind: str | None,
) -> None:
    simulator = FakeSimulator(step_results=((False, False, raw_info),))
    harness = make_env(simulators=(simulator,))
    try:
        harness.env.reset()
        _, _, _, _, info = harness.env.step(DrivingAction.KEEP)
        context = harness.reward.contexts[-1]
        assert context.arrived is arrived
        assert context.collision_kind == collision_kind
        assert all(info[key] is value for key, value in raw_info.items())
    finally:
        harness.env.close()


def test_agent_failure_uses_conservative_analysis_and_does_not_crash_rollout() -> None:
    harness = make_env(suite=FailingSuite(), shield=SafetyShield(make_config().shield))
    try:
        initial_observation, _ = harness.env.reset()
        observation, reward, terminated, truncated, info = harness.env.step(DrivingAction.KEEP)

        assert harness.env.observation_space.contains(initial_observation)
        assert harness.env.observation_space.contains(observation)
        assert harness.env_factory.created[0].actions == [int(DrivingAction.STOP)]
        assert info["executed_action"] == int(DrivingAction.STOP)
        assert info["decision_trace"].claims == ()
        assert info["decision_trace"].review.reasons == ("agent_analysis_failed",)
        assert np.isfinite(reward)
        assert terminated is False
        assert truncated is False
    finally:
        harness.env.close()


def test_post_step_observation_failure_returns_safe_values_and_truncates() -> None:
    observation_builder = RecordingObservationBuilder(failing_calls=(2,))
    harness = make_env(
        shield=SafetyShield(make_config().shield),
        observation=observation_builder,
    )
    try:
        initial_observation, _ = harness.env.reset()
        observation, _, terminated, truncated, info = harness.env.step(DrivingAction.KEEP)

        assert harness.env.observation_space.contains(initial_observation)
        assert np.isfinite(initial_observation).all()
        assert harness.env_factory.created[0].actions == [int(DrivingAction.KEEP)]
        assert harness.env.observation_space.contains(observation)
        assert np.isfinite(observation).all()
        assert terminated is False
        assert truncated is True
        assert "observation_error" in info
    finally:
        harness.env.close()


def test_reward_exception_after_simulator_advance_closes_and_recreates() -> None:
    failed = FakeSimulator()
    replacement = FakeSimulator()
    reward = RecordingRewardCalculator(failing_calculate_calls=(1,))
    harness = make_env(simulators=(failed, replacement), reward=reward)
    try:
        harness.env.reset(seed=90)

        with pytest.raises(RuntimeError, match="reward calculate failed"):
            harness.env.step(DrivingAction.KEEP)

        assert failed.actions == [int(DrivingAction.STOP)]
        assert failed.close_calls == 1
        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)

        observation, info = harness.env.reset(seed=91)
        assert harness.env_factory.created == [failed, replacement]
        assert replacement.reset_seeds == [91]
        assert harness.env.observation_space.contains(observation)
        assert info["seed"] == 91
        harness.env.step(DrivingAction.KEEP)
        assert replacement.actions == [int(DrivingAction.STOP)]
    finally:
        harness.env.close()


def test_reward_context_exception_after_simulator_advance_closes_and_recreates() -> None:
    failed = FakeSimulator()
    failed.config["decision_repeat"] = 0
    replacement = FakeSimulator()
    harness = make_env(simulators=(failed, replacement))
    try:
        harness.env.reset(seed=92)

        with pytest.raises(ValueError, match="decision interval must be positive"):
            harness.env.step(DrivingAction.KEEP)

        assert failed.actions == [int(DrivingAction.STOP)]
        assert failed.close_calls == 1
        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)

        observation, _ = harness.env.reset(seed=93)
        assert harness.env.observation_space.contains(observation)
        assert replacement.reset_seeds == [93]
        harness.env.step(DrivingAction.KEEP)
        assert replacement.actions == [int(DrivingAction.STOP)]
    finally:
        harness.env.close()


def test_trace_exception_after_simulator_advance_closes_and_recreates() -> None:
    failed = FakeSimulator()
    replacement = FakeSimulator()
    reward = RecordingRewardCalculator(invalid_trace_calls=(1,))
    harness = make_env(simulators=(failed, replacement), reward=reward)
    try:
        harness.env.reset(seed=94)

        with pytest.raises(ValueError, match="reward_components must be finite"):
            harness.env.step(DrivingAction.KEEP)

        assert failed.actions == [int(DrivingAction.STOP)]
        assert failed.close_calls == 1
        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)

        observation, _ = harness.env.reset(seed=95)
        assert harness.env.observation_space.contains(observation)
        assert replacement.reset_seeds == [95]
        harness.env.step(DrivingAction.KEEP)
        assert replacement.actions == [int(DrivingAction.STOP)]
    finally:
        harness.env.close()


def test_simulator_step_exception_truncates_closes_and_recreates_on_reset() -> None:
    failed = FakeSimulator(fail_on_step=True)
    replacement = FakeSimulator()
    harness = make_env(simulators=(failed, replacement))
    try:
        harness.env.reset(seed=10)
        observation, reward, terminated, truncated, info = harness.env.step(DrivingAction.KEEP)

        assert harness.env.observation_space.contains(observation)
        assert np.isfinite(observation).all()
        assert reward == 0.0
        assert terminated is False
        assert truncated is True
        assert info["simulator_error"] == "simulator step failed"
        assert failed.close_calls == 1

        reset_observation, reset_info = harness.env.reset(seed=11)
        assert harness.env_factory.created == [failed, replacement]
        assert replacement.reset_seeds == [11]
        assert harness.env.observation_space.contains(reset_observation)
        assert reset_info["seed"] == 11
    finally:
        harness.env.close()
