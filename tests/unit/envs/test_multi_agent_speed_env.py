from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray
from stable_baselines3.common.vec_env import DummyVecEnv

import mad_driving.envs.multi_agent_speed_env as multi_agent_speed_env_module
from mad_driving.agents.critic import CriticAgent
from mad_driving.agents.nominal import NominalMotionAgent
from mad_driving.agents.noop_critic import NoOpCritic
from mad_driving.agents.suite import AgentAnalysisResult, AgentSuite
from mad_driving.config.models import AppConfig, ControlConfig, ShieldConfig
from mad_driving.control import DrivingAction
from mad_driving.envs import MultiAgentSpeedEnv
from mad_driving.envs.reward import RewardContext, RewardResult
from mad_driving.interfaces import (
    CriticReview,
    DecisionTrace,
    RiskClaim,
    RoadContext,
    SceneFrame,
    SceneObservation,
    ShieldResult,
)
from mad_driving.methods import get_method_profile
from mad_driving.safety import SafetyShield
from mad_driving.scenarios import (
    EpisodeSeedAllocator,
    EpisodeSeeds,
    NoOpScenarioRuntime,
    ScenarioObservationContext,
    ScenarioState,
    ScenarioStepResult,
)
from mad_driving.scenarios.runtime import ScenarioTransition
from mad_driving.world_model import SceneSnapshotBuilder
from tests.unit.agents.factories import make_analysis, make_claim, make_frame, make_snapshot


class FakeLane:
    speed_limit = 54.0
    index = ("A", "B", 0)
    width = 3.5

    @staticmethod
    def local_coordinates(position: tuple[float, float]) -> tuple[float, float]:
        return position


class FakeNavigation:
    def __init__(self) -> None:
        self.current_lane = FakeLane()
        self.route_completion = 0.25
        self.travelled_length = 12.5


class FakeVehicle:
    LENGTH = 4.5
    WIDTH = 1.8

    def __init__(
        self,
        name: str = "ego",
        *,
        position: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.name = name
        self.position = position
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
    def __init__(
        self,
        vehicle: FakeVehicle,
        actors: Sequence[FakeVehicle] = (),
    ) -> None:
        self._objects = {vehicle.name: vehicle}
        self._objects.update({actor.name: actor for actor in actors})

    def get_objects(self) -> dict[str, FakeVehicle]:
        return dict(self._objects)


class FakeSimulator:
    def __init__(
        self,
        *,
        step_results: Sequence[tuple[bool, bool, dict[str, Any]]] = (),
        actors: Sequence[FakeVehicle] = (),
        actual_seed_offset: int = 0,
        fail_on_reset_calls: Sequence[int] = (),
        fail_on_step: bool = False,
        fail_on_close: bool = False,
        physics_dt_after_reset: float | None = None,
        physics_dt_after_step: float | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.vehicle = FakeVehicle()
        self.agent = self.vehicle
        self.engine = FakeEngine(self.vehicle, actors)
        self.action_space = gym.spaces.Discrete(4)
        self.config: dict[str, Any] = {
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
            "map_config": {"lane_width": 3.5},
        }
        self.step_results = tuple(step_results)
        self.actual_seed_offset = actual_seed_offset
        self.fail_on_reset_calls = set(fail_on_reset_calls)
        self.fail_on_step = fail_on_step
        self.fail_on_close = fail_on_close
        self.physics_dt_after_reset = physics_dt_after_reset
        self.physics_dt_after_step = physics_dt_after_step
        self.events = events
        self.reset_seeds: list[int | None] = []
        self.actions: list[int] = []
        self.close_calls = 0
        self.current_seed: int | None = None
        self.reset_info: dict[str, object] = {"simulator_reset": True}

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, object], dict[str, object]]:
        self.reset_seeds.append(seed)
        if self.events is not None:
            self.events.append("simulator.reset")
        if len(self.reset_seeds) in self.fail_on_reset_calls:
            raise RuntimeError("simulator failure")
        if seed is None:
            raise AssertionError("the outer environment must select a simulator scenario")
        self.current_seed = seed + self.actual_seed_offset
        self.actions.clear()
        self.vehicle.position = (0.0, 0.0)
        self.vehicle.navigation.route_completion = 0.25
        self.vehicle.crash_vehicle = False
        self.vehicle.crash_human = False
        info = dict(self.reset_info)
        info["env_seed"] = self.current_seed
        if self.physics_dt_after_reset is not None:
            self.config["physics_world_step_size"] = self.physics_dt_after_reset
        return {}, info

    def step(self, action: int) -> tuple[dict[str, object], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        if self.events is not None:
            self.events.append("simulator.step")
        if self.fail_on_step:
            raise RuntimeError("simulator failure")
        index = len(self.actions) - 1
        terminated, truncated, info = (
            self.step_results[index] if index < len(self.step_results) else (False, False, {})
        )
        self.vehicle.position = (float(len(self.actions)), 0.0)
        self.vehicle.navigation.route_completion += 0.1
        self.vehicle.crash_vehicle = bool(info.get("crash_vehicle", False))
        self.vehicle.crash_human = bool(info.get("crash_human", False))
        if self.physics_dt_after_step is not None:
            self.config["physics_world_step_size"] = self.physics_dt_after_step
        return {}, 999.0, terminated, truncated, info

    def close(self) -> None:
        self.close_calls += 1
        if self.events is not None:
            self.events.append("simulator.close")
        if self.fail_on_close:
            raise RuntimeError("simulator close failure")


class RecordingEnvironmentFactory:
    def __init__(
        self,
        simulators: Sequence[FakeSimulator] = (),
        *,
        fail_on_calls: Sequence[int] = (),
        runtime_config_overrides: Mapping[str, object] | None = None,
    ) -> None:
        self._simulators = list(simulators)
        self._fail_on_calls = set(fail_on_calls)
        self._runtime_config_overrides = dict(runtime_config_overrides or {})
        self.created: list[FakeSimulator] = []
        self.calls: list[tuple[dict[str, object], ControlConfig]] = []

    def __call__(
        self,
        options: dict[str, object],
        control_config: ControlConfig,
    ) -> FakeSimulator:
        self.calls.append((options, control_config))
        if len(self.calls) in self._fail_on_calls:
            raise RuntimeError("environment creation failure")
        simulator = self._simulators.pop(0) if self._simulators else FakeSimulator()
        simulator.config.update(options)
        simulator.config.update(self._runtime_config_overrides)
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
            raise RuntimeError(f"{self.name} factory failure")
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
    def __init__(self, results: Sequence[AgentAnalysisResult] = ()) -> None:
        self.results = tuple(results) or (make_analysis(claims=complete_claims()),)
        self.observations: list[SceneObservation] = []

    def analyze(self, observation: SceneObservation) -> AgentAnalysisResult:
        self.observations.append(observation)
        index = min(len(self.observations) - 1, len(self.results) - 1)
        return self.results[index]


class FailingSuite:
    def analyze(self, observation: SceneObservation) -> AgentAnalysisResult:
        del observation
        raise RuntimeError("analysis failure")


class RecordingRuntime:
    def __init__(
        self,
        *,
        events: list[str] | None = None,
        context: ScenarioObservationContext | None = None,
        step_results: Sequence[ScenarioStepResult] = (),
        invalid_reset_result: bool = False,
        invalid_after_reset_result: bool = False,
        invalid_before_step_result: bool = False,
        invalid_step_result: bool = False,
        invalid_transition_state: bool = False,
        invalid_transition_outcome: bool = False,
        invalid_context_result: bool = False,
        reset_state: ScenarioState | None = None,
        after_reset_state: ScenarioState | None = None,
        before_step_state: ScenarioState | None = None,
        transition_state: ScenarioState | None = None,
        physics_dt_after_simulator_reset: float | None = None,
        physics_dt_after_step: float | None = None,
    ) -> None:
        self.events = events
        self.context = context or ScenarioObservationContext(
            scenario_id="unit_multi_agent_speed_env"
        )
        self.step_results = tuple(step_results) or (ScenarioStepResult(False, False),)
        self.invalid_reset_result = invalid_reset_result
        self.invalid_after_reset_result = invalid_after_reset_result
        self.invalid_before_step_result = invalid_before_step_result
        self.invalid_step_result = invalid_step_result
        self.invalid_transition_state = invalid_transition_state
        self.invalid_transition_outcome = invalid_transition_outcome
        self.invalid_context_result = invalid_context_result
        self.reset_state = reset_state
        self.after_reset_state = after_reset_state
        self.before_step_state = before_step_state
        self.transition_state = transition_state
        self.physics_dt_after_simulator_reset = physics_dt_after_simulator_reset
        self.physics_dt_after_step = physics_dt_after_step
        self.states: list[ScenarioState] = []
        self.after_step_calls = 0

    def reset(self, environment: object, *, seeds: EpisodeSeeds) -> ScenarioState:
        del environment
        if self.events is not None:
            self.events.append("runtime.reset")
        if self.invalid_reset_result:
            return object()  # type: ignore[return-value]
        state = self.reset_state or ScenarioState(self.context.scenario_id, seeds, {})
        self.states.append(state)
        return state

    def after_simulator_reset(self, environment: object, state: ScenarioState) -> ScenarioState:
        if self.events is not None:
            self.events.append("runtime.after_simulator_reset")
        if self.physics_dt_after_simulator_reset is not None:
            simulator = cast(FakeSimulator, environment)
            simulator.config["physics_world_step_size"] = self.physics_dt_after_simulator_reset
        if self.invalid_after_reset_result:
            return object()  # type: ignore[return-value]
        return self.after_reset_state or state

    def before_step(
        self,
        environment: object,
        state: ScenarioState,
        *,
        step_index: int,
    ) -> ScenarioState:
        del environment, step_index
        if self.events is not None:
            self.events.append("runtime.before_step")
        if self.invalid_before_step_result:
            return object()  # type: ignore[return-value]
        return self.before_step_state or state

    def after_step(
        self,
        environment: object,
        state: ScenarioState,
        *,
        step_index: int,
        raw_info: Mapping[str, object],
    ) -> ScenarioTransition:
        del step_index, raw_info
        if self.events is not None:
            self.events.append("runtime.after_step")
        if self.physics_dt_after_step is not None:
            simulator = cast(FakeSimulator, environment)
            simulator.config["physics_world_step_size"] = self.physics_dt_after_step
        if self.invalid_step_result:
            return object()  # type: ignore[return-value]
        index = min(self.after_step_calls, len(self.step_results) - 1)
        self.after_step_calls += 1
        transition_state = self.transition_state or state
        outcome = self.step_results[index]
        if self.invalid_transition_state:
            transition_state = object()  # type: ignore[assignment]
        if self.invalid_transition_outcome:
            outcome = object()  # type: ignore[assignment]
        return ScenarioTransition(state=transition_state, outcome=outcome)  # type: ignore[arg-type]

    def observation_context(self, state: ScenarioState) -> ScenarioObservationContext:
        del state
        if self.invalid_context_result:
            return object()  # type: ignore[return-value]
        return self.context


class StateThreadingRuntime:
    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id
        self.before_step_inputs: list[ScenarioState] = []
        self.after_step_inputs: list[ScenarioState] = []

    def _state(self, seeds: EpisodeSeeds, phase: str) -> ScenarioState:
        return ScenarioState(self.scenario_id, seeds, {"phase": phase})

    def reset(self, environment: object, *, seeds: EpisodeSeeds) -> ScenarioState:
        del environment
        return self._state(seeds, "reset")

    def after_simulator_reset(self, environment: object, state: ScenarioState) -> ScenarioState:
        del environment
        return self._state(state.seeds, "after_simulator_reset")

    def before_step(
        self,
        environment: object,
        state: ScenarioState,
        *,
        step_index: int,
    ) -> ScenarioState:
        del environment
        self.before_step_inputs.append(state)
        return self._state(state.seeds, f"before_step_{step_index}")

    def after_step(
        self,
        environment: object,
        state: ScenarioState,
        *,
        step_index: int,
        raw_info: Mapping[str, object],
    ) -> ScenarioTransition:
        del environment, raw_info
        self.after_step_inputs.append(state)
        return ScenarioTransition(
            state=self._state(state.seeds, f"after_step_{step_index}"),
            outcome=ScenarioStepResult(False, False),
        )

    def observation_context(self, state: ScenarioState) -> ScenarioObservationContext:
        phase = str(state.parameters["phase"])
        distances = {
            "after_simulator_reset": 10.0,
            "after_step_1": 9.0,
            "after_step_2": 8.0,
        }
        return ScenarioObservationContext(
            scenario_id=state.scenario_id,
            distance_to_conflict_point_m=distances[phase],
        )


class RecordingSnapshotBuilder:
    def __init__(
        self,
        *,
        failing_calls: Sequence[int] = (),
        events: list[str] | None = None,
    ) -> None:
        self.failing_calls = set(failing_calls)
        self.events = events
        self.calls: list[dict[str, Any]] = []

    def build(
        self,
        env: FakeSimulator,
        *,
        step_index: int,
        seeds: EpisodeSeeds,
        context: ScenarioObservationContext,
        scenario_result: ScenarioStepResult,
        raw_info: Mapping[str, object],
        previous_executed_action: int,
        previous_shield_intervention: bool,
    ) -> SceneFrame:
        call = {
            "env": env,
            "step_index": step_index,
            "seeds": seeds,
            "context": context,
            "scenario_result": scenario_result,
            "raw_info": dict(raw_info),
            "previous_executed_action": previous_executed_action,
            "previous_shield_intervention": previous_shield_intervention,
        }
        self.calls.append(call)
        if self.events is not None:
            self.events.append("frame.initial" if step_index == 0 else "frame.next")
        if len(self.calls) in self.failing_calls:
            raise RuntimeError("snapshot failure")
        observation = make_snapshot(
            step_index=step_index,
            occlusion_regions=context.occlusion_regions,
            road_context=RoadContext(
                stop_required=context.stop_required,
                distance_to_conflict_point_m=context.distance_to_conflict_point_m,
                intersection_entry_prohibited=context.intersection_entry_prohibited,
            ),
            previous_executed_action=previous_executed_action,
            previous_shield_intervention=previous_shield_intervention,
            ego_speed_mps=env.vehicle.speed,
            speed_limit_mps=FakeLane.speed_limit / 3.6,
        )
        collision_kind = None
        if bool(raw_info.get("crash_human", False)):
            collision_kind = "crossing_actor"
        elif bool(raw_info.get("crash_vehicle", False)):
            collision_kind = "vehicle"
        return make_frame(
            scenario_id=context.scenario_id,
            seeds=seeds,
            observation=observation,
            collision_occurred=collision_kind is not None,
            collision_kind=collision_kind,
            off_road=bool(raw_info.get("out_of_road", False)),
            arrived=bool(raw_info.get("arrive_dest", False)),
            scenario_success=scenario_result.success,
            scenario_failure=scenario_result.failure,
        )


class RecordingObservationBuilder:
    def __init__(self, *, failing_calls: Sequence[int] = ()) -> None:
        self.failing_calls = set(failing_calls)
        self.calls: list[tuple[SceneObservation, tuple[RiskClaim, ...], CriticReview]] = []

    def build(
        self,
        observation: SceneObservation,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> NDArray[np.float32]:
        self.calls.append((observation, tuple(claims), review))
        if len(self.calls) in self.failing_calls:
            raise RuntimeError("observation failure")
        value = np.float32(min(observation.step_index / 10.0, 1.0))
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
            raise RuntimeError("reward reset failure")

    def calculate(self, context: RewardContext) -> RewardResult:
        self.calculate_calls += 1
        self.contexts.append(context)
        if self.calculate_calls in self.failing_calculate_calls:
            raise RuntimeError("reward failure")
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
        self.calls: list[
            tuple[
                DrivingAction | int,
                SceneObservation,
                tuple[RiskClaim, ...],
                tuple[str, ...],
                tuple[str, ...],
            ]
        ] = []

    def filter(
        self,
        requested_action: DrivingAction | int,
        observation: SceneObservation,
        claims: Sequence[RiskClaim],
        *,
        expected_agent_ids: Sequence[str],
        failed_agent_ids: Sequence[str],
    ) -> ShieldResult:
        self.calls.append(
            (
                requested_action,
                observation,
                tuple(claims),
                tuple(expected_agent_ids),
                tuple(failed_agent_ids),
            )
        )
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


def make_config_for_scenario(scenario_id: str) -> AppConfig:
    values = make_config().model_dump(mode="python")
    values["scenario_id"] = scenario_id
    return AppConfig.model_validate(values)


def make_config_for_method(method_id: str, *, shield_mode: str = "monitor") -> AppConfig:
    values = make_config().model_dump(mode="python")
    values["method"] = {"id": method_id}
    values["shield"]["mode"] = shield_mode
    return AppConfig.model_validate(values)


@dataclass
class EnvHarness:
    env: MultiAgentSpeedEnv
    env_factory: RecordingEnvironmentFactory
    suite: SequenceSuite | FailingSuite
    shield: RecordingShield | SafetyShield
    snapshot_builder: RecordingSnapshotBuilder | SceneSnapshotBuilder
    reward: RecordingRewardCalculator
    observation: RecordingObservationBuilder
    runtime: RecordingRuntime | NoOpScenarioRuntime


def make_env(
    *,
    config: AppConfig | None = None,
    role: str = "train",
    worker_index: int = 0,
    simulators: Sequence[FakeSimulator] = (),
    env_factory: RecordingEnvironmentFactory | None = None,
    suite: SequenceSuite | FailingSuite | None = None,
    shield: RecordingShield | SafetyShield | None = None,
    snapshot_builder: RecordingSnapshotBuilder | SceneSnapshotBuilder | None = None,
    reward: RecordingRewardCalculator | None = None,
    observation: RecordingObservationBuilder | None = None,
    runtime: RecordingRuntime | NoOpScenarioRuntime | None = None,
    suite_factory: Callable[..., Any] | None = None,
    shield_factory: Callable[..., Any] | None = None,
    builder_factory: Callable[..., Any] | None = None,
    reward_factory: Callable[..., Any] | None = None,
    observation_factory: Callable[..., Any] | None = None,
    runtime_factory: Callable[..., Any] | None = None,
) -> EnvHarness:
    selected_config = config or make_config()
    selected_env_factory = env_factory or RecordingEnvironmentFactory(simulators)
    selected_suite = suite or SequenceSuite()
    selected_shield = shield or RecordingShield()
    selected_snapshot_builder = snapshot_builder or RecordingSnapshotBuilder()
    selected_reward = reward or RecordingRewardCalculator()
    selected_observation = observation or RecordingObservationBuilder()
    selected_runtime = runtime or RecordingRuntime()
    env = MultiAgentSpeedEnv(
        selected_config,
        role=role,  # type: ignore[arg-type]
        worker_index=worker_index,
        scenario_runtime_factory=runtime_factory or (lambda scenario_id: selected_runtime),
        env_factory=selected_env_factory,
        suite_factory=suite_factory or (lambda agents_config: selected_suite),
        shield_factory=shield_factory or (lambda shield_config: selected_shield),
        builder_factory=builder_factory or (lambda: selected_snapshot_builder),
        reward_factory=reward_factory or (lambda reward_config: selected_reward),
        observation_factory=observation_factory
        or (lambda observation_config: selected_observation),
    )
    return EnvHarness(
        env=env,
        env_factory=selected_env_factory,
        suite=selected_suite,
        shield=selected_shield,
        snapshot_builder=selected_snapshot_builder,
        reward=selected_reward,
        observation=selected_observation,
        runtime=selected_runtime,
    )


def make_default_composition_env(
    config: AppConfig,
    *,
    shield_factory: Callable[..., Any] | None = None,
) -> MultiAgentSpeedEnv:
    simulator = FakeSimulator()
    arguments: dict[str, Any] = {}
    if shield_factory is not None:
        arguments["shield_factory"] = shield_factory
    return MultiAgentSpeedEnv(
        config,
        role="train",
        worker_index=0,
        scenario_runtime_factory=lambda scenario_id: RecordingRuntime(),
        env_factory=RecordingEnvironmentFactory((simulator,)),
        builder_factory=RecordingSnapshotBuilder,
        reward_factory=lambda reward_config: RecordingRewardCalculator(),
        observation_factory=lambda observation_config: RecordingObservationBuilder(),
        **arguments,
    )


@pytest.mark.parametrize(
    "method_id",
    (
        "b0_rule",
        "b1_nominal",
        "b2_multi_no_review",
        "proposed",
        "proposed_no_critic",
        "proposed_no_shield",
        "proposed_no_hazard",
    ),
)
def test_default_environment_composes_the_selected_method_profile(method_id: str) -> None:
    config = make_config_for_method(method_id)
    profile = get_method_profile(config.method.id)
    env = make_default_composition_env(config)
    try:
        env.reset(seed=17)

        suite = cast(AgentSuite, env._suite)
        shield = cast(SafetyShield, env._shield)
        assert suite.expected_agent_ids == profile.specialist_ids
        assert isinstance(suite.critic, CriticAgent) is profile.critic_enabled
        assert isinstance(suite.critic, NoOpCritic) is not profile.critic_enabled
        assert shield._config.mode == profile.default_shield_mode
    finally:
        env.close()


def test_default_suite_routes_expected_failed_agents_to_the_shield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_nominal(
        self: NominalMotionAgent, observation: SceneObservation
    ) -> tuple[RiskClaim, ...]:
        del self, observation
        raise RuntimeError("injected nominal failure")

    monkeypatch.setattr(NominalMotionAgent, "analyze", fail_nominal)
    shield = RecordingShield()
    env = make_default_composition_env(
        make_config_for_method("proposed_no_hazard"),
        shield_factory=lambda shield_config: shield,
    )
    try:
        env.reset(seed=18)
        env.step(DrivingAction.KEEP)

        assert shield.calls[0][3] == ("nominal", "rule")
        assert shield.calls[0][4] == ("nominal",)
    finally:
        env.close()


def expected_seeds(
    episode_rng_seed: int,
    *,
    role: str = "train",
    worker_index: int = 0,
    config: AppConfig | None = None,
) -> EpisodeSeeds:
    selected_config = config or make_config()
    split = getattr(selected_config.scenarios, role)
    return EpisodeSeedAllocator(role, split, worker_index).allocate(episode_rng_seed)  # type: ignore[arg-type]


def assert_episode_metadata(
    info: Mapping[str, object],
    seeds: EpisodeSeeds,
    *,
    role: str = "train",
    worker_index: int = 0,
) -> None:
    assert info["episode_rng_seed"] == seeds.episode_rng_seed
    assert info["environment_seed"] == seeds.episode_rng_seed
    assert info["simulator_seed"] == seeds.metadrive_scenario_index
    assert info["scenario_seed"] == seeds.scenario_parameter_seed
    assert info["metadrive_scenario_index"] == seeds.metadrive_scenario_index
    assert info["scenario_selection_seed"] == seeds.scenario_selection_seed
    assert info["scenario_parameter_seed"] == seeds.scenario_parameter_seed
    assert info["role"] == role
    assert info["worker_index"] == worker_index
    for key in (
        "episode_rng_seed",
        "environment_seed",
        "simulator_seed",
        "scenario_seed",
        "metadrive_scenario_index",
        "scenario_selection_seed",
        "scenario_parameter_seed",
        "role",
        "worker_index",
    ):
        assert type(info[key]) in {int, str}


def test_env_exposes_fixed_spaces_and_explicit_reset_is_reproducible() -> None:
    harness = make_env()
    try:
        first, info = harness.env.reset(seed=123)
        second, second_info = harness.env.reset(seed=123)
        seeds = expected_seeds(123)

        assert harness.env.action_space == gym.spaces.Discrete(4)
        assert harness.env.observation_space.shape == (24,)
        assert harness.env.observation_space.dtype == np.float32
        assert harness.env.observation_space.contains(first)
        np.testing.assert_array_equal(first, second)
        assert_episode_metadata(info, seeds)
        assert_episode_metadata(second_info, seeds)
        assert harness.env_factory.created[0].reset_seeds == [
            seeds.metadrive_scenario_index,
            seeds.metadrive_scenario_index,
        ]
        assert harness.reward.reset_calls == 2
    finally:
        harness.env.close()


def test_implicit_resets_advance_reproducible_episode_seed_sequence() -> None:
    first = make_env()
    second = make_env()
    try:
        first_values = [first.env.reset(seed=42)[1]["episode_rng_seed"]]
        second_values = [second.env.reset(seed=42)[1]["episode_rng_seed"]]
        first_values.extend(first.env.reset()[1]["episode_rng_seed"] for _ in range(3))
        second_values.extend(second.env.reset()[1]["episode_rng_seed"] for _ in range(3))
        assert first_values == second_values
        assert len(set(first_values)) == len(first_values)
    finally:
        first.env.close()
        second.env.close()


def test_first_implicit_reset_uses_config_seed_and_later_implicit_reset_advances() -> None:
    harness = make_env()
    try:
        _, first_info = harness.env.reset()
        _, second_info = harness.env.reset()
        assert first_info["episode_rng_seed"] == 42
        assert second_info["episode_rng_seed"] != 42
    finally:
        harness.env.close()


def test_explicit_reseed_restarts_the_implicit_sequence() -> None:
    harness = make_env()
    try:
        harness.env.reset(seed=91)
        first_sequence = [harness.env.reset()[1]["episode_rng_seed"] for _ in range(3)]
        harness.env.reset(seed=91)
        second_sequence = [harness.env.reset()[1]["episode_rng_seed"] for _ in range(3)]
        assert first_sequence == second_sequence
    finally:
        harness.env.close()


def test_vecenv_auto_reset_does_not_reuse_initial_episode_seed() -> None:
    runtime = RecordingRuntime()
    simulator = FakeSimulator(step_results=((False, True, {"max_step": True}),))
    harness = make_env(simulators=(simulator,), runtime=runtime)
    vector = DummyVecEnv([lambda: harness.env])
    try:
        vector.seed(42)
        vector.reset()
        vector.step(np.asarray([int(DrivingAction.KEEP)]))
        assert runtime.states[0].seeds.episode_rng_seed == 42
        assert len({state.seeds.episode_rng_seed for state in runtime.states[:2]}) == 2
    finally:
        vector.close()


def test_runtime_hook_order_wraps_simulator_and_frame_construction() -> None:
    events: list[str] = []
    runtime = RecordingRuntime(events=events)
    simulator = FakeSimulator(events=events)
    builder = RecordingSnapshotBuilder(events=events)
    harness = make_env(simulators=(simulator,), runtime=runtime, snapshot_builder=builder)
    try:
        harness.env.reset(seed=7)
        assert events == [
            "runtime.reset",
            "simulator.reset",
            "runtime.after_simulator_reset",
            "frame.initial",
        ]

        harness.env.step(DrivingAction.KEEP)
        assert events == [
            "runtime.reset",
            "simulator.reset",
            "runtime.after_simulator_reset",
            "frame.initial",
            "runtime.before_step",
            "simulator.step",
            "runtime.after_step",
            "frame.next",
        ]
    finally:
        harness.env.close()


def test_runtime_hook_states_drive_context_and_persist_between_steps() -> None:
    runtime = StateThreadingRuntime(make_config().scenario_id)
    harness = make_env(runtime=runtime)  # type: ignore[arg-type]
    try:
        harness.env.reset(seed=7)
        assert (
            harness.snapshot_builder.calls[-1][  # type: ignore[union-attr]
                "context"
            ].distance_to_conflict_point_m
            == 10.0
        )

        harness.env.step(DrivingAction.KEEP)
        harness.env.step(DrivingAction.KEEP)

        assert runtime.after_step_inputs[0].parameters["phase"] == "before_step_1"
        assert runtime.before_step_inputs[1].parameters["phase"] == "after_step_1"
        assert (
            harness.snapshot_builder.calls[-1][  # type: ignore[union-attr]
                "context"
            ].distance_to_conflict_point_m
            == 8.0
        )
    finally:
        harness.env.close()


def test_reset_and_step_publish_complete_episode_metadata() -> None:
    harness = make_env(role="validation", worker_index=2)
    try:
        _, reset_info = harness.env.reset(seed=71)
        seeds = expected_seeds(71, role="validation", worker_index=2)
        assert_episode_metadata(reset_info, seeds, role="validation", worker_index=2)

        _, _, _, _, step_info = harness.env.step(DrivingAction.KEEP)
        assert_episode_metadata(step_info, seeds, role="validation", worker_index=2)
        assert harness.runtime.states[0].seeds == seeds  # type: ignore[union-attr]
        assert harness.snapshot_builder.calls[0]["seeds"] == seeds  # type: ignore[union-attr]
    finally:
        harness.env.close()


def test_environment_uses_role_split_for_the_simulator_scenario_range() -> None:
    harness = make_env(role="validation", worker_index=3)
    try:
        harness.env.reset(seed=72)
        options = harness.env_factory.calls[0][0]
        assert options["start_seed"] == 10_000
        assert options["num_scenarios"] == 1_000
    finally:
        harness.env.close()


def test_actual_simulator_scenario_mismatch_closes_and_raises() -> None:
    simulator = FakeSimulator(actual_seed_offset=1)
    harness = make_env(simulators=(simulator,))
    with pytest.raises(RuntimeError, match="scenario index mismatch"):
        harness.env.reset(seed=73)
    assert simulator.close_calls == 1
    with pytest.raises(RuntimeError, match="fatally closed"):
        harness.env.reset(seed=74)
    harness.env.close()
    assert simulator.close_calls == 1


def test_runtime_rejects_stale_prior_episode_state_before_simulator_reset() -> None:
    simulator = FakeSimulator()
    stale_state = ScenarioState(
        scenario_id=make_config().scenario_id,
        seeds=expected_seeds(70),
        parameters={},
    )
    runtime = RecordingRuntime(reset_state=stale_state)
    harness = make_env(simulators=(simulator,), runtime=runtime)

    with pytest.raises(RuntimeError, match="ScenarioState seeds mismatch"):
        harness.env.reset(seed=71)

    assert simulator.reset_seeds == []
    assert simulator.close_calls == 1


@pytest.mark.parametrize(
    ("state_scenario_id", "context_scenario_id", "message"),
    [
        ("stale_scenario", "unit_multi_agent_speed_env", "ScenarioState scenario_id"),
        ("unit_multi_agent_speed_env", "stale_context", "observation context scenario_id"),
    ],
)
def test_runtime_rejects_inconsistent_scenario_identity(
    state_scenario_id: str,
    context_scenario_id: str,
    message: str,
) -> None:
    seeds = expected_seeds(72)
    runtime = RecordingRuntime(
        context=ScenarioObservationContext(scenario_id=context_scenario_id),
        reset_state=ScenarioState(state_scenario_id, seeds, {}),
    )
    harness = make_env(runtime=runtime)

    with pytest.raises(RuntimeError, match=message):
        harness.env.reset(seed=72)

    assert harness.env_factory.created[0].close_calls == 1


@pytest.mark.parametrize(
    "runtime_factory",
    [lambda scenario_id: object()],
)
def test_scenario_runtime_factory_return_type_is_validated(
    runtime_factory: Callable[..., object],
) -> None:
    harness = make_env(runtime_factory=runtime_factory)
    with pytest.raises(TypeError, match="ScenarioRuntime"):
        harness.env.reset(seed=74)
    assert harness.env_factory.created[0].close_calls == 1


@pytest.mark.parametrize(
    ("runtime", "operation", "message"),
    [
        (RecordingRuntime(invalid_reset_result=True), "reset", "ScenarioState"),
        (
            RecordingRuntime(invalid_after_reset_result=True),
            "reset",
            "after_simulator_reset.*ScenarioState",
        ),
        (RecordingRuntime(invalid_context_result=True), "reset", "ScenarioObservationContext"),
        (
            RecordingRuntime(invalid_before_step_result=True),
            "step",
            "before_step.*ScenarioState",
        ),
        (RecordingRuntime(invalid_step_result=True), "step", "ScenarioTransition"),
        (
            RecordingRuntime(invalid_transition_state=True),
            "step",
            "ScenarioTransition.state",
        ),
        (
            RecordingRuntime(invalid_transition_outcome=True),
            "step",
            "ScenarioTransition.outcome",
        ),
    ],
)
def test_scenario_runtime_hook_return_types_are_validated(
    runtime: RecordingRuntime,
    operation: str,
    message: str,
) -> None:
    harness = make_env(runtime=runtime)
    if operation == "reset":
        with pytest.raises(TypeError, match=message):
            harness.env.reset(seed=75)
    else:
        harness.env.reset(seed=75)
        with pytest.raises(TypeError, match=message):
            harness.env.step(DrivingAction.KEEP)
    assert harness.env_factory.created[0].close_calls == 1


@pytest.mark.parametrize(
    ("hook", "mismatch", "message"),
    [
        ("after_simulator_reset", "scenario_id", "ScenarioState scenario_id"),
        ("after_simulator_reset", "seeds", "ScenarioState seeds"),
        ("before_step", "scenario_id", "ScenarioState scenario_id"),
        ("before_step", "seeds", "ScenarioState seeds"),
        ("after_step", "scenario_id", "ScenarioState scenario_id"),
        ("after_step", "seeds", "ScenarioState seeds"),
    ],
)
def test_runtime_validates_scenario_identity_after_every_state_hook(
    hook: str,
    mismatch: str,
    message: str,
) -> None:
    seeds = expected_seeds(76)
    returned_seeds = EpisodeSeeds(1, 2, 3, 4) if mismatch == "seeds" else seeds
    scenario_id = "stale_scenario" if mismatch == "scenario_id" else make_config().scenario_id
    returned_state = ScenarioState(scenario_id, returned_seeds, {})
    runtime_kwargs = {
        "after_simulator_reset": {"after_reset_state": returned_state},
        "before_step": {"before_step_state": returned_state},
        "after_step": {"transition_state": returned_state},
    }[hook]
    harness = make_env(runtime=RecordingRuntime(**runtime_kwargs))

    if hook == "after_simulator_reset":
        with pytest.raises(RuntimeError, match=message):
            harness.env.reset(seed=76)
    else:
        harness.env.reset(seed=76)
        with pytest.raises(RuntimeError, match=message):
            harness.env.step(DrivingAction.KEEP)

    assert harness.env_factory.created[0].close_calls == 1


def test_default_noop_runtime_has_no_extra_simulator_side_effects() -> None:
    simulator = FakeSimulator()
    noop = NoOpScenarioRuntime(make_config().scenario_id)
    harness = make_env(simulators=(simulator,), runtime=noop)
    try:
        harness.env.reset(seed=76)
        harness.env.step(DrivingAction.KEEP)
        assert len(simulator.reset_seeds) == 1
        assert simulator.actions == [int(DrivingAction.STOP)]
    finally:
        harness.env.close()


def test_runtime_context_flows_to_frame_builder_and_hidden_actor_stays_privileged() -> None:
    visible = FakeVehicle("visible", position=(5.0, 0.0))
    hidden = FakeVehicle("hidden", position=(8.0, 0.0))
    simulator = FakeSimulator(actors=(hidden, visible))
    context = ScenarioObservationContext(
        scenario_id="visibility_case",
        stop_required=True,
        distance_to_conflict_point_m=12.5,
        intersection_entry_prohibited=True,
        visible_actor_ids=frozenset({"visible"}),
    )
    runtime = RecordingRuntime(context=context)
    suite = SequenceSuite()
    harness = make_env(
        config=make_config_for_scenario("visibility_case"),
        simulators=(simulator,),
        runtime=runtime,
        suite=suite,
        snapshot_builder=SceneSnapshotBuilder(),
    )
    try:
        harness.env.reset(seed=77)
        observation = suite.observations[0]
        assert tuple(actor.actor_id for actor in observation.visible_actors) == ("visible",)
        assert not hasattr(observation, "scenario_id")
        assert not hasattr(observation, "seeds")
        assert observation.road_context.stop_required is True
        assert observation.road_context.distance_to_conflict_point_m == 12.5
        assert observation.road_context.intersection_entry_prohibited is True
        assert "hidden" not in repr(observation)
    finally:
        harness.env.close()


def test_step_routes_agent_status_to_shield_but_excludes_it_from_reward_context() -> None:
    pre = make_analysis(claims=complete_claims(severity=0.1))
    post = make_analysis(claims=complete_claims(severity=0.9))
    suite = SequenceSuite((pre, post))
    shield = RecordingShield(DrivingAction.STOP)
    harness = make_env(suite=suite, shield=shield)
    try:
        harness.env.reset(seed=78)
        harness.env.step(DrivingAction.KEEP)

        assert shield.calls[0][1] is suite.observations[0]
        assert not isinstance(shield.calls[0][1], SceneFrame)
        assert shield.calls[0][3] == pre.expected_agent_ids
        assert shield.calls[0][4] == pre.failed_agent_ids
        context = harness.reward.contexts[-1]
        assert context.previous_frame.observation is suite.observations[0]
        assert context.next_frame.observation is suite.observations[1]
        assert not hasattr(context, "previous_analysis")
        assert not hasattr(context, "next_analysis")
        assert context.executed_action == int(DrivingAction.STOP)
        assert context.decision_interval_s == pytest.approx(0.10)
    finally:
        harness.env.close()


def test_runtime_timing_mismatch_fails_reset_before_simulator_and_initial_frame() -> None:
    events: list[str] = []
    simulator = FakeSimulator(events=events, fail_on_close=True)
    runtime = RecordingRuntime(events=events)
    builder = RecordingSnapshotBuilder(events=events)
    env_factory = RecordingEnvironmentFactory(
        (simulator,),
        runtime_config_overrides={"physics_world_step_size": 0.03},
    )
    harness = make_env(
        env_factory=env_factory,
        runtime=runtime,
        snapshot_builder=builder,
    )

    with pytest.raises(
        RuntimeError,
        match="decision interval mismatch.*0.15.*0.1",
    ) as captured:
        harness.env.reset(seed=78)

    assert simulator.reset_seeds == []
    assert builder.calls == []
    assert events == ["runtime.reset", "simulator.close"]
    assert simulator.close_calls == 1
    assert captured.value.__notes__ == ["simulator cleanup failed: simulator close failure"]


def test_simulator_reset_timing_mutation_fails_before_initial_frame() -> None:
    events: list[str] = []
    simulator = FakeSimulator(
        events=events,
        fail_on_close=True,
        physics_dt_after_reset=0.03,
    )
    runtime = RecordingRuntime(events=events)
    builder = RecordingSnapshotBuilder(events=events)
    harness = make_env(
        simulators=(simulator,),
        runtime=runtime,
        snapshot_builder=builder,
    )

    with pytest.raises(
        RuntimeError,
        match="decision interval mismatch.*0.15.*0.1",
    ) as captured:
        harness.env.reset(seed=78)

    assert simulator.reset_seeds == [expected_seeds(78).metadrive_scenario_index]
    assert builder.calls == []
    assert events == [
        "runtime.reset",
        "simulator.reset",
        "runtime.after_simulator_reset",
        "simulator.close",
    ]
    assert simulator.close_calls == 1
    assert captured.value.__notes__ == ["simulator cleanup failed: simulator close failure"]


def test_after_simulator_reset_timing_mutation_fails_before_initial_frame() -> None:
    events: list[str] = []
    simulator = FakeSimulator(events=events)
    runtime = RecordingRuntime(
        events=events,
        physics_dt_after_simulator_reset=0.03,
    )
    builder = RecordingSnapshotBuilder(events=events)
    harness = make_env(
        simulators=(simulator,),
        runtime=runtime,
        snapshot_builder=builder,
    )

    with pytest.raises(RuntimeError, match="decision interval mismatch.*0.15.*0.1"):
        harness.env.reset(seed=78)

    assert builder.calls == []
    assert events == [
        "runtime.reset",
        "simulator.reset",
        "runtime.after_simulator_reset",
        "simulator.close",
    ]
    assert simulator.close_calls == 1


def test_timing_change_after_reset_fails_before_transition_side_effects() -> None:
    events: list[str] = []
    simulator = FakeSimulator(events=events)
    runtime = RecordingRuntime(events=events)
    builder = RecordingSnapshotBuilder(events=events)
    harness = make_env(
        simulators=(simulator,),
        runtime=runtime,
        snapshot_builder=builder,
    )
    harness.env.reset(seed=78)
    events.clear()
    simulator.config["physics_world_step_size"] = 0.03

    with pytest.raises(RuntimeError, match="decision interval mismatch.*0.15.*0.1"):
        harness.env.step(DrivingAction.KEEP)

    assert events == ["simulator.close"]
    assert simulator.actions == []
    assert runtime.after_step_calls == 0
    assert len(builder.calls) == 1
    assert harness.reward.calculate_calls == 0
    assert simulator.close_calls == 1


def test_simulator_step_timing_mutation_fails_before_post_step_consumers() -> None:
    events: list[str] = []
    simulator = FakeSimulator(
        events=events,
        fail_on_close=True,
        physics_dt_after_step=0.03,
    )
    runtime = RecordingRuntime(events=events)
    builder = RecordingSnapshotBuilder(events=events)
    suite = SequenceSuite()
    harness = make_env(
        simulators=(simulator,),
        runtime=runtime,
        snapshot_builder=builder,
        suite=suite,
    )
    harness.env.reset(seed=78)
    events.clear()

    with pytest.raises(
        RuntimeError,
        match="decision interval mismatch.*0.15.*0.1",
    ) as captured:
        harness.env.step(DrivingAction.KEEP)

    assert events == [
        "runtime.before_step",
        "simulator.step",
        "simulator.close",
    ]
    assert runtime.after_step_calls == 0
    assert len(builder.calls) == 1
    assert len(suite.observations) == 1
    assert harness.reward.calculate_calls == 0
    assert simulator.close_calls == 1
    assert captured.value.__notes__ == ["simulator cleanup failed: simulator close failure"]
    with pytest.raises(RuntimeError, match="reset"):
        harness.env.step(DrivingAction.KEEP)


def test_runtime_after_step_timing_mutation_fails_before_next_frame() -> None:
    events: list[str] = []
    simulator = FakeSimulator(events=events)
    runtime = RecordingRuntime(events=events, physics_dt_after_step=0.03)
    builder = RecordingSnapshotBuilder(events=events)
    suite = SequenceSuite()
    harness = make_env(
        simulators=(simulator,),
        runtime=runtime,
        snapshot_builder=builder,
        suite=suite,
    )
    harness.env.reset(seed=78)
    events.clear()

    with pytest.raises(RuntimeError, match="decision interval mismatch.*0.15.*0.1"):
        harness.env.step(DrivingAction.KEEP)

    assert events == [
        "runtime.before_step",
        "simulator.step",
        "runtime.after_step",
        "simulator.close",
    ]
    assert runtime.after_step_calls == 1
    assert len(builder.calls) == 1
    assert len(suite.observations) == 1
    assert harness.reward.calculate_calls == 0
    assert simulator.close_calls == 1
    with pytest.raises(RuntimeError, match="reset"):
        harness.env.step(DrivingAction.KEEP)


@pytest.mark.parametrize("failure", ["simulator", "snapshot", "reward", "observation"])
def test_internal_failure_closes_simulator_once_and_propagates(failure: str) -> None:
    simulator = FakeSimulator(fail_on_step=failure == "simulator")
    builder = RecordingSnapshotBuilder(failing_calls=(2,) if failure == "snapshot" else ())
    reward = RecordingRewardCalculator(failing_calculate_calls=(1,) if failure == "reward" else ())
    observation = RecordingObservationBuilder(
        failing_calls=(2,) if failure == "observation" else ()
    )
    harness = make_env(
        simulators=(simulator,),
        snapshot_builder=builder,
        reward=reward,
        observation=observation,
    )
    harness.env.reset(seed=79)

    with pytest.raises(RuntimeError, match=failure):
        harness.env.step(DrivingAction.KEEP)
    assert simulator.close_calls == 1
    harness.env.close()
    assert simulator.close_calls == 1


def test_cleanup_failure_is_noted_without_masking_primary_exception() -> None:
    simulator = FakeSimulator(fail_on_step=True, fail_on_close=True)
    harness = make_env(simulators=(simulator,))
    harness.env.reset(seed=80)

    with pytest.raises(RuntimeError, match="simulator failure") as captured:
        harness.env.step(DrivingAction.KEEP)
    assert captured.value.__notes__ == ["simulator cleanup failed: simulator close failure"]
    assert simulator.close_calls == 1


@pytest.mark.parametrize(
    ("raw_terminated", "raw_truncated", "raw_info", "scenario_result"),
    [
        (False, True, {}, ScenarioStepResult(False, False)),
        (True, False, {"crash_vehicle": True}, ScenarioStepResult(False, False)),
        (True, True, {}, ScenarioStepResult(True, False)),
    ],
)
def test_raw_gymnasium_flags_are_preserved_when_termination_agrees_with_typed_outcome(
    raw_terminated: bool,
    raw_truncated: bool,
    raw_info: dict[str, bool],
    scenario_result: ScenarioStepResult,
) -> None:
    simulator = FakeSimulator(step_results=((raw_terminated, raw_truncated, raw_info),))
    runtime = RecordingRuntime(step_results=(scenario_result,))
    harness = make_env(simulators=(simulator,), runtime=runtime)
    try:
        harness.env.reset(seed=81)
        _, _, terminated, truncated, _ = harness.env.step(DrivingAction.KEEP)
        assert terminated is raw_terminated
        assert truncated is raw_truncated
    finally:
        harness.env.close()


def test_unmatched_raw_termination_is_a_fatal_consistency_error() -> None:
    simulator = FakeSimulator(step_results=((True, False, {}),), fail_on_close=True)
    harness = make_env(simulators=(simulator,))
    harness.env.reset(seed=81)

    with pytest.raises(
        RuntimeError,
        match="raw simulator termination.*typed privileged outcome",
    ) as captured:
        harness.env.step(DrivingAction.KEEP)

    assert simulator.close_calls == 1
    assert captured.value.__notes__ == ["simulator cleanup failed: simulator close failure"]
    with pytest.raises(RuntimeError, match="reset"):
        harness.env.step(DrivingAction.KEEP)


@pytest.mark.parametrize(
    ("raw_info", "scenario_result"),
    [
        ({"crash_vehicle": True}, ScenarioStepResult(False, False)),
        ({}, ScenarioStepResult(True, False)),
        ({}, ScenarioStepResult(False, True)),
    ],
)
def test_collision_success_and_failure_are_terminated(
    raw_info: dict[str, bool],
    scenario_result: ScenarioStepResult,
) -> None:
    simulator = FakeSimulator(step_results=((False, False, raw_info),))
    runtime = RecordingRuntime(step_results=(scenario_result,))
    harness = make_env(simulators=(simulator,), runtime=runtime)
    try:
        harness.env.reset(seed=82)
        _, _, terminated, truncated, _ = harness.env.step(DrivingAction.KEEP)
        assert terminated is True
        assert truncated is False
    finally:
        harness.env.close()


@pytest.mark.parametrize(
    ("raw_info", "expected_collision"),
    [
        ({"crash_vehicle": True}, True),
        ({}, False),
    ],
)
def test_step_info_exposes_typed_boolean_collision_for_curriculum(
    raw_info: dict[str, bool],
    expected_collision: bool,
) -> None:
    simulator = FakeSimulator(step_results=((False, False, raw_info),))
    harness = make_env(simulators=(simulator,))
    try:
        harness.env.reset(seed=82)
        _, _, _, _, info = harness.env.step(DrivingAction.KEEP)

        assert info["collision_occurred"] is expected_collision
        assert type(info["collision_occurred"]) is bool
    finally:
        harness.env.close()


def test_privileged_off_road_terminates_without_raw_termination() -> None:
    simulator = FakeSimulator(step_results=((False, False, {"out_of_road": True}),))
    harness = make_env(simulators=(simulator,))
    try:
        harness.env.reset(seed=82)
        _, _, terminated, truncated, _ = harness.env.step(DrivingAction.KEEP)
        assert terminated is True
        assert truncated is False
    finally:
        harness.env.close()


def test_horizon_and_collision_can_be_truncated_and_terminated_together() -> None:
    simulator = FakeSimulator(
        step_results=((False, True, {"max_step": True, "crash_vehicle": True}),)
    )
    harness = make_env(simulators=(simulator,))
    try:
        harness.env.reset(seed=83)
        _, _, terminated, truncated, _ = harness.env.step(DrivingAction.KEEP)
        assert terminated is True
        assert truncated is True
    finally:
        harness.env.close()


def test_reset_clears_episode_local_reward_action_frame_and_runtime_state() -> None:
    runtime = RecordingRuntime()
    harness = make_env(runtime=runtime)
    try:
        harness.env.reset(seed=84)
        harness.env.step(DrivingAction.KEEP)
        harness.env.reset(seed=85)

        assert harness.reward.reset_calls == 2
        assert len(runtime.states) == 2
        assert runtime.states[0] is not runtime.states[1]
        initial_call = harness.snapshot_builder.calls[-1]  # type: ignore[union-attr]
        assert initial_call["step_index"] == 0
        assert initial_call["previous_executed_action"] == int(DrivingAction.KEEP)
        assert initial_call["previous_shield_intervention"] is False
    finally:
        harness.env.close()


def test_returned_info_and_reward_components_do_not_alias_internal_values() -> None:
    raw_step_info: dict[str, Any] = {
        "simulator_value": "original",
        "nested": {"values": [1, {"status": "original"}]},
    }
    simulator = FakeSimulator(step_results=((False, False, raw_step_info),))
    harness = make_env(simulators=(simulator,))
    try:
        _, reset_info = harness.env.reset(seed=86)
        reset_info["simulator_reset"] = False
        assert simulator.reset_info == {"simulator_reset": True}

        _, _, _, _, info = harness.env.step(DrivingAction.KEEP)
        info["simulator_value"] = "changed"
        nested = info["nested"]
        assert isinstance(nested, dict)
        values = nested["values"]
        assert isinstance(values, list)
        assert isinstance(values[1], dict)
        values[1]["status"] = "changed"
        assert raw_step_info == {
            "simulator_value": "original",
            "nested": {"values": [1, {"status": "original"}]},
        }

        components = info["reward_components"]
        trace = info["decision_trace"]
        assert isinstance(components, dict)
        assert isinstance(trace, DecisionTrace)
        components["progress_reward"] = 99.0
        assert trace.reward_components["progress_reward"] == 1.25
        assert harness.reward.components["progress_reward"] == 1.25
    finally:
        harness.env.close()


def test_info_tracks_route_progress_and_safe_unnecessary_stop_duration() -> None:
    simulator = FakeSimulator()
    harness = make_env(simulators=(simulator,))
    try:
        harness.env.reset(seed=84)

        _, _, _, _, first_info = harness.env.step(DrivingAction.STOP)
        _, _, _, _, second_info = harness.env.step(DrivingAction.STOP)

        assert first_info["route_progress"] == pytest.approx(0.25)
        assert first_info["unnecessary_stop_duration_s"] == pytest.approx(0.1)
        assert second_info["unnecessary_stop_duration_s"] == pytest.approx(0.2)

        harness.env.reset(seed=85)
        _, _, _, _, reset_episode_info = harness.env.step(DrivingAction.STOP)
        assert reset_episode_info["unnecessary_stop_duration_s"] == pytest.approx(0.1)
    finally:
        harness.env.close()


def test_step_builds_trace_from_pre_step_analysis_and_post_step_reward() -> None:
    pre = make_analysis(
        claims=(make_claim("nominal"), make_claim("rule")),
        failed_agent_ids=("hazard",),
        errors=("hazard:RuntimeError:pre-decision failure",),
    )
    post = make_analysis(
        claims=complete_claims(severity=0.9),
        review=neutral_review(reason="post"),
    )
    suite = SequenceSuite((pre, post))
    shield = RecordingShield(DrivingAction.STOP)
    harness = make_env(suite=suite, shield=shield)
    try:
        harness.env.reset(seed=87)
        observation, reward, terminated, truncated, info = harness.env.step(DrivingAction.KEEP)

        simulator = harness.env_factory.created[0]
        assert shield.calls[0][0] == DrivingAction.KEEP
        assert shield.calls[0][2] == pre.claims
        assert simulator.actions == [int(DrivingAction.STOP)]
        assert harness.observation.calls[-1][1] == post.claims
        trace = info["decision_trace"]
        assert trace.claims == pre.claims
        assert trace.review == pre.review
        assert trace.failed_agent_ids == pre.failed_agent_ids
        assert trace.errors == pre.errors
        assert info["failed_agent_ids"] == pre.failed_agent_ids
        assert info["analysis_errors"] == pre.errors
        assert trace.reward_components == info["reward_components"]
        assert trace.required_action == info["required_action"] == DrivingAction.STOP
        assert trace.intervention_required is info["intervention_required"] is True
        assert trace.episode_rng_seed == 87
        assert trace.metadrive_scenario_index == info["metadrive_scenario_index"]
        assert trace.scenario_selection_seed == info["scenario_selection_seed"]
        assert trace.scenario_parameter_seed == info["scenario_parameter_seed"]
        assert trace.role == "train"
        assert trace.worker_index == 0
        assert trace.scenario_id == "unit_multi_agent_speed_env"
        assert trace.difficulty_level == 0
        assert reward == pytest.approx(sum(info["reward_components"].values()))
        assert harness.env.observation_space.contains(observation)
        assert terminated is False
        assert truncated is False
    finally:
        harness.env.close()


def test_low_level_control_fail_safe_is_a_fatal_internal_error() -> None:
    class MonitorShield:
        def filter(
            self,
            requested_action: DrivingAction | int,
            observation: SceneObservation,
            claims: Sequence[RiskClaim],
            *,
            expected_agent_ids: Sequence[str],
            failed_agent_ids: Sequence[str],
        ) -> ShieldResult:
            del observation, claims, expected_agent_ids, failed_agent_ids
            requested = DrivingAction(requested_action)
            return ShieldResult(
                requested_action=requested,
                required_action=DrivingAction.STOP,
                executed_action=requested,
                intervention_required=True,
                intervened=False,
                reasons=("imminent_ttc",),
            )

    simulator = FakeSimulator(
        step_results=(
            (
                False,
                False,
                {"fail_safe": True, "fail_safe_reason": "ValueError"},
            ),
        )
    )
    harness = make_env(simulators=(simulator,), shield=MonitorShield())
    harness.env.reset(seed=123)

    with pytest.raises(RuntimeError, match="low-level control fail-safe: ValueError"):
        harness.env.step(DrivingAction.KEEP)

    assert simulator.close_calls == 1
    with pytest.raises(RuntimeError, match="fatally closed"):
        harness.env.reset(seed=124)


def test_per_agent_failure_result_remains_a_valid_mdp_step() -> None:
    failed = make_analysis(
        claims=(make_claim("nominal"), make_claim("rule")),
        failed_agent_ids=("hazard",),
        errors=("hazard:RuntimeError:failed",),
    )
    harness = make_env(
        suite=SequenceSuite((failed, failed)),
        shield=SafetyShield(make_config().shield),
    )
    try:
        initial_observation, _ = harness.env.reset(seed=88)
        observation, reward, terminated, truncated, info = harness.env.step(DrivingAction.KEEP)
        assert harness.env.observation_space.contains(initial_observation)
        assert harness.env.observation_space.contains(observation)
        assert info["decision_trace"].claims == failed.claims
        assert np.isfinite(reward)
        assert terminated is False
        assert truncated is False
    finally:
        harness.env.close()


def test_trace_records_pre_step_analysis_and_shield_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter((100, 2_000_100, 5_000_000, 5_500_000, 8_000_000, 11_000_000))
    monkeypatch.setattr(
        multi_agent_speed_env_module.time,
        "perf_counter_ns",
        lambda: next(clock_values),
    )
    harness = make_env()
    try:
        harness.env.reset(seed=89)
        _, _, _, _, info = harness.env.step(DrivingAction.KEEP)
        trace = info["decision_trace"]

        assert trace.expected_agent_ids == ("nominal", "hazard", "rule")
        assert trace.analysis_latency_ms == pytest.approx(2.0)
        assert trace.shield_latency_ms == pytest.approx(0.5)
        assert np.isfinite(trace.analysis_latency_ms)
        assert np.isfinite(trace.shield_latency_ms)
        assert trace.analysis_latency_ms >= 0.0
        assert trace.shield_latency_ms >= 0.0
    finally:
        harness.env.close()


def test_latency_differences_do_not_change_environment_scientific_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(clock_samples: tuple[int, ...]) -> tuple[np.ndarray, float, DecisionTrace]:
        clock_values = iter(clock_samples)
        monkeypatch.setattr(
            multi_agent_speed_env_module.time,
            "perf_counter_ns",
            lambda: next(clock_values),
        )
        harness = make_env()
        try:
            harness.env.reset(seed=90)
            observation, reward, _, _, info = harness.env.step(DrivingAction.KEEP)
            return observation, reward, info["decision_trace"]  # type: ignore[return-value]
        finally:
            harness.env.close()

    first = run((0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000))
    second = run((0, 9_000_000, 10_000_000, 12_000_000, 13_000_000, 16_000_000))

    first_record = asdict(first[2])
    second_record = asdict(second[2])
    for record in (first_record, second_record):
        record.pop("analysis_latency_ms")
        record.pop("shield_latency_ms")

    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert first_record == second_record
    assert (
        first[2].analysis_latency_ms,
        first[2].shield_latency_ms,
    ) != (
        second[2].analysis_latency_ms,
        second[2].shield_latency_ms,
    )


def test_default_oracle_ignores_claims_and_intentionally_disabled_specialists() -> None:
    def reset_with_method(method_id: str) -> tuple[float | None, tuple[str, ...]]:
        config_values = make_config_for_method(method_id).model_dump(mode="python")
        config_values["agents"]["hazard"].update(  # type: ignore[index]
            {"reaction_delay_s": 0.25, "ego_max_safe_deceleration_mps2": -5.0}
        )
        actor = FakeVehicle("hidden-lead", position=(12.0, 0.0))
        actor.velocity = (5.0, 0.0)
        actor.last_velocity = (5.0, 0.0)
        runtime = RecordingRuntime(
            context=ScenarioObservationContext(
                scenario_id="unit_multi_agent_speed_env",
                visible_actor_ids=frozenset(),
            )
        )
        environment = MultiAgentSpeedEnv(
            AppConfig.model_validate(config_values),
            role="train",
            worker_index=0,
            scenario_runtime_factory=lambda scenario_id: runtime,
            env_factory=RecordingEnvironmentFactory((FakeSimulator(actors=(actor,)),)),
            reward_factory=lambda reward_config: RecordingRewardCalculator(),
            observation_factory=lambda observation_config: RecordingObservationBuilder(),
        )
        try:
            environment.reset(seed=91)
            assert environment._frame is not None
            assert environment._analysis is not None
            assert environment._frame.observation.visible_actors == ()
            return (
                environment._frame.privileged.minimum_actual_stopping_margin_m,
                environment._analysis.expected_agent_ids,
            )
        finally:
            environment.close()

    proposed_margin, proposed_agents = reset_with_method("proposed")
    ablated_margin, ablated_agents = reset_with_method("proposed_no_hazard")

    assert proposed_agents == ("nominal", "hazard", "rule")
    assert ablated_agents == ("nominal", "rule")
    assert proposed_margin == ablated_margin == pytest.approx(2.5)


def test_suite_level_analysis_error_is_fatal() -> None:
    simulator = FakeSimulator()
    harness = make_env(simulators=(simulator,), suite=FailingSuite())
    with pytest.raises(RuntimeError, match="analysis failure"):
        harness.env.reset(seed=89)
    assert simulator.close_calls == 1


def test_trace_construction_error_is_fatal() -> None:
    simulator = FakeSimulator()
    reward = RecordingRewardCalculator(invalid_trace_calls=(1,))
    harness = make_env(simulators=(simulator,), reward=reward)
    harness.env.reset(seed=90)
    with pytest.raises(ValueError, match="reward_components must be finite"):
        harness.env.step(DrivingAction.KEEP)
    assert simulator.close_calls == 1


def test_reward_context_validation_error_is_fatal() -> None:
    simulator = FakeSimulator()
    harness = make_env(simulators=(simulator,))
    harness.env.reset(seed=91)
    simulator.config["decision_repeat"] = 0
    with pytest.raises(ValueError, match="decision interval must be positive"):
        harness.env.step(DrivingAction.KEEP)
    assert simulator.close_calls == 1


def test_close_propagates_simulator_failure_and_remains_idempotent() -> None:
    simulator = FakeSimulator(fail_on_close=True)
    harness = make_env(simulators=(simulator,))
    harness.env.reset(seed=92)
    with pytest.raises(RuntimeError, match="simulator close failure"):
        harness.env.close()
    harness.env.close()
    assert simulator.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        harness.env.reset(seed=93)


def test_step_before_reset_and_after_terminal_transition_are_rejected() -> None:
    simulator = FakeSimulator(step_results=((False, True, {"max_step": True}),))
    harness = make_env(simulators=(simulator,))
    try:
        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)
        harness.env.reset(seed=94)
        harness.env.step(DrivingAction.KEEP)
        with pytest.raises(RuntimeError, match="reset"):
            harness.env.step(DrivingAction.KEEP)
    finally:
        harness.env.close()


def test_reset_and_step_return_exact_gymnasium_runtime_types() -> None:
    harness = make_env(shield=RecordingShield(DrivingAction.KEEP))
    try:
        reset_result = harness.env.reset(seed=95)
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


def test_numpy_and_python_integer_actions_are_accepted() -> None:
    harness = make_env(shield=RecordingShield(DrivingAction.KEEP))
    try:
        harness.env.reset(seed=96)
        sampled_action = harness.env.action_space.sample()
        assert isinstance(sampled_action, np.integer)
        harness.env.step(sampled_action)
        harness.env.reset(seed=97)
        harness.env.step(1)
        assert harness.env_factory.created[0].actions == [1]
    finally:
        harness.env.close()


@pytest.mark.parametrize("action", [True, False, 1.0, np.float32(1.0), -1, 4])
def test_invalid_actions_are_rejected_without_clipping(action: object) -> None:
    harness = make_env()
    try:
        harness.env.reset(seed=98)
        with pytest.raises(ValueError, match="action"):
            harness.env.step(action)  # type: ignore[arg-type]
        assert harness.env_factory.created[0].actions == []
    finally:
        harness.env.close()


@pytest.mark.parametrize(("role", "worker_index"), [("invalid", 0), ("train", -1)])
def test_constructor_validates_role_and_worker_index(role: str, worker_index: int) -> None:
    with pytest.raises(ValueError, match="role|worker_index"):
        make_env(role=role, worker_index=worker_index)


def test_constructor_requires_explicit_role_and_worker_index() -> None:
    with pytest.raises(TypeError, match="role.*worker_index|worker_index.*role"):
        MultiAgentSpeedEnv(make_config())  # type: ignore[call-arg]


def test_simulator_creation_is_lazy() -> None:
    env_factory = RecordingEnvironmentFactory()
    harness = make_env(env_factory=env_factory)
    assert env_factory.calls == []
    harness.env.close()


def test_environment_lifecycle_protocols_are_public() -> None:
    from mad_driving.envs import DrivingEnvironment, ScenarioRuntimeFactory

    assert DrivingEnvironment.__name__ == "DrivingEnvironment"
    assert ScenarioRuntimeFactory.__name__ == "ScenarioRuntimeFactory"


def test_dependency_factory_failure_closes_owned_simulator_and_is_fatal() -> None:
    simulator = FakeSimulator()
    harness = make_env(
        simulators=(simulator,),
        runtime_factory=ConfigurableFactory("runtime", object(), fail_on_calls=(1,)),
    )
    with pytest.raises(RuntimeError, match="runtime factory failure"):
        harness.env.reset(seed=99)
    assert simulator.close_calls == 1
    with pytest.raises(RuntimeError, match="fatally closed"):
        harness.env.reset(seed=100)


class DifficultyRecordingRuntimeFactory:
    def __init__(self, runtime: RecordingRuntime) -> None:
        self.runtime = runtime
        self.levels: list[int] = []

    def __call__(self, scenario_id: str) -> RecordingRuntime:
        del scenario_id
        return self.runtime

    def set_difficulty_level(self, level: int) -> None:
        self.levels.append(level)


class EvaluationScheduleRecordingRuntimeFactory(DifficultyRecordingRuntimeFactory):
    def __init__(self, runtime: RecordingRuntime) -> None:
        super().__init__(runtime)
        self.schedules: list[tuple[str, ...]] = []

    def set_scenario_schedule(self, scenario_ids: tuple[str, ...]) -> None:
        self.schedules.append(scenario_ids)


def test_difficulty_level_is_forwarded_to_the_scenario_factory() -> None:
    runtime = RecordingRuntime()
    factory = DifficultyRecordingRuntimeFactory(runtime)
    harness = make_env(runtime=runtime, runtime_factory=factory)

    harness.env.set_difficulty_level(2)

    assert factory.levels == [2]


@pytest.mark.parametrize("role", ["validation", "test"])
def test_evaluation_schedule_is_forwarded_for_non_training_roles(role: str) -> None:
    runtime = RecordingRuntime()
    factory = EvaluationScheduleRecordingRuntimeFactory(runtime)
    harness = make_env(role=role, runtime=runtime, runtime_factory=factory)

    harness.env.set_evaluation_scenario_schedule(("lead_brake", "cut_in"))

    assert factory.schedules == [("lead_brake", "cut_in")]


def test_training_environment_rejects_evaluation_schedule_installation() -> None:
    runtime = RecordingRuntime()
    factory = EvaluationScheduleRecordingRuntimeFactory(runtime)
    harness = make_env(role="train", runtime=runtime, runtime_factory=factory)

    with pytest.raises(ValueError, match="validation.*test"):
        harness.env.set_evaluation_scenario_schedule(("lead_brake",))

    assert factory.schedules == []


@pytest.mark.parametrize("mode", ["monitor", "enforce", "off"])
@pytest.mark.parametrize("role", ["validation", "test"])
def test_evaluation_shield_mode_is_used_to_construct_the_next_episode(mode: str, role: str) -> None:
    shield_configs: list[ShieldConfig] = []

    def shield_factory(shield_config: ShieldConfig) -> RecordingShield:
        shield_configs.append(shield_config)
        return RecordingShield()

    harness = make_env(role=role, shield_factory=shield_factory)

    harness.env.set_evaluation_shield_mode(mode)  # type: ignore[arg-type]
    harness.env.reset(seed=10_001 if role == "validation" else 20_001)

    assert [value.mode for value in shield_configs] == [mode]
    assert harness.env.observation_space.shape == (24,)


def test_training_environment_rejects_evaluation_shield_mode_installation() -> None:
    harness = make_env(role="train")

    with pytest.raises(ValueError, match="validation.*test"):
        harness.env.set_evaluation_shield_mode("monitor")


def test_active_evaluation_episode_rejects_shield_mode_mutation() -> None:
    harness = make_env(role="test")
    harness.env.reset(seed=20_001)

    with pytest.raises(RuntimeError, match="active"):
        harness.env.set_evaluation_shield_mode("off")


def test_evaluation_scene_read_requires_active_episode_and_returns_current_visible_view() -> None:
    harness = make_env(role="test")

    with pytest.raises(RuntimeError, match="reset"):
        harness.env.current_scene_observation_for_evaluation()

    harness.env.reset(seed=20_001)
    initial = harness.env.current_scene_observation_for_evaluation()
    harness.env.step(int(DrivingAction.KEEP))
    current = harness.env.current_scene_observation_for_evaluation()

    assert isinstance(initial, SceneObservation)
    assert initial.step_index == 0
    assert current.step_index == 1
    assert not hasattr(current, "privileged")


def test_reset_and_step_info_include_scenario_metadata() -> None:
    harness = make_env(runtime=RecordingRuntime())
    _, reset_info = harness.env.reset(seed=42)
    _, _, _, _, step_info = harness.env.step(int(DrivingAction.KEEP))

    for info in (reset_info, step_info):
        assert info["scenario_id"] == "unit_multi_agent_speed_env"
        assert info["difficulty_level"] == 0
        assert info["scenario_parameters"] == {}
        assert info["scenario_success"] is False
        assert info["scenario_failure"] is False


def test_step_info_exposes_strict_evaluation_record_telemetry_without_changing_shape() -> None:
    harness = make_env(shield=RecordingShield(DrivingAction.KEEP))
    harness.env.reset(seed=42)

    observation, _, _, _, info = harness.env.step(int(DrivingAction.KEEP))

    assert observation.shape == (24,)
    assert info["simulation_time_s"] == pytest.approx(0.1)
    assert info["decision_interval_s"] == pytest.approx(0.1)
    assert info["ego_speed_mps"] == pytest.approx(10.0)
    assert info["ego_longitudinal_acceleration_mps2"] == pytest.approx(0.0)
    assert info["route_completion"] == pytest.approx(0.25)
    assert info["route_progress_m"] == pytest.approx(12.5)
    assert info["lane_offset_m"] == pytest.approx(0.0)
    assert info["collision_kind"] is None
    assert info["minimum_actual_ttc_s"] is None
    assert info["minimum_actual_stopping_margin_m"] is None
    assert info["pre_step_hard_rule_constraint"] is False
    assert info["post_step_rule_violation_event"] is False
    assert info["arrived"] is False
    assert info["off_road"] is False
