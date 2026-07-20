"""Simulator lifecycle boundaries and the concrete Gymnasium environment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from mad_driving.agents.suite import (
    AgentAnalysisResult,
    AgentSuite,
    AnalysisSuite,
    SuiteFactory,
    analyze_safely,
)
from mad_driving.config.models import (
    AppConfig,
    ControlConfig,
    ObservationConfig,
    RewardConfig,
    ShieldConfig,
)
from mad_driving.control import DrivingAction, target_speed_mps
from mad_driving.coordinator import ObservationBuilder
from mad_driving.envs.reward import RewardCalculator, RewardContext, RewardResult
from mad_driving.interfaces import (
    CriticReview,
    DecisionTrace,
    RiskClaim,
    SceneFrame,
    SceneObservation,
    SceneSnapshot,
    ShieldResult,
)
from mad_driving.safety import SafetyShield
from mad_driving.scenarios import (
    EnvironmentRole,
    EpisodeSeedAllocator,
    EpisodeSeeds,
    NoOpScenarioRuntime,
    ScenarioObservationContext,
    ScenarioRuntime,
    ScenarioState,
    ScenarioStepResult,
)
from mad_driving.world_model import SceneSnapshotBuilder
from mad_driving.world_model.validation import decision_interval_s


class DrivingEnvironment(Protocol):
    """Subset of the MetaDrive API owned by the outer Gymnasium environment."""

    config: dict[str, Any]
    vehicle: Any
    engine: Any
    agent: Any
    action_space: Any
    current_seed: int

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]: ...

    def step(
        self, action: tuple[float, float] | int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...


class EnvironmentFactory(Protocol):
    def __call__(self, config: dict[str, object]) -> DrivingEnvironment: ...


class ControlEnvironmentFactory(Protocol):
    def __call__(
        self,
        config: dict[str, object],
        control_config: ControlConfig,
    ) -> DrivingEnvironment: ...


class Shield(Protocol):
    def filter(
        self,
        requested_action: DrivingAction | int,
        observation: SceneObservation,
        claims: Sequence[RiskClaim],
    ) -> ShieldResult: ...


class ShieldFactory(Protocol):
    def __call__(self, config: ShieldConfig) -> Shield: ...


class FrameBuilder(Protocol):
    def build(
        self,
        env: DrivingEnvironment,
        *,
        step_index: int,
        seeds: EpisodeSeeds,
        context: ScenarioObservationContext,
        scenario_result: ScenarioStepResult,
        raw_info: Mapping[str, object],
        previous_executed_action: int,
        previous_shield_intervention: bool,
    ) -> SceneFrame: ...


class FrameBuilderFactory(Protocol):
    def __call__(self) -> FrameBuilder: ...


class Reward(Protocol):
    def reset(self) -> None: ...

    def calculate(self, context: RewardContext) -> RewardResult: ...


class RewardFactory(Protocol):
    def __call__(self, config: RewardConfig) -> Reward: ...


class Observation(Protocol):
    def build(
        self,
        observation: SceneObservation,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> NDArray[np.float32]: ...


class ObservationFactory(Protocol):
    def __call__(self, config: ObservationConfig) -> Observation: ...


class ScenarioRuntimeFactory(Protocol):
    def __call__(self, scenario_id: str) -> ScenarioRuntime: ...


@dataclass(frozen=True)
class SmokeResult:
    """Finite summary of one fixed-action headless episode."""

    steps_completed: int
    terminated: bool
    truncated: bool
    final_snapshot: SceneSnapshot
    final_claims: tuple[RiskClaim, ...]
    final_review: CriticReview


@dataclass(frozen=True)
class ControlSmokeResult:
    """Finite summary of one shielded four-action control episode."""

    steps_completed: int
    terminated: bool
    truncated: bool
    final_snapshot: SceneSnapshot
    final_claims: tuple[RiskClaim, ...]
    final_review: CriticReview
    final_trace: DecisionTrace
    action_counts: tuple[int, int, int, int]
    shield_intervention_count: int


def _create_control_environment(
    config: dict[str, object],
    control_config: ControlConfig,
) -> DrivingEnvironment:
    from mad_driving.envs.control_metadrive_env import create_control_metadrive_env

    return create_control_metadrive_env(config, control_config)


class MultiAgentSpeedEnv(gym.Env[NDArray[np.float32], int]):
    """Gymnasium wrapper around the deterministic multi-Agent speed pipeline."""

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        config: AppConfig,
        *,
        role: EnvironmentRole = "train",
        worker_index: int = 0,
        scenario_runtime_factory: ScenarioRuntimeFactory = NoOpScenarioRuntime,
        env_factory: ControlEnvironmentFactory = _create_control_environment,
        suite_factory: SuiteFactory = AgentSuite.from_config,
        shield_factory: ShieldFactory = SafetyShield,
        builder_factory: FrameBuilderFactory = SceneSnapshotBuilder,
        reward_factory: RewardFactory = RewardCalculator,
        observation_factory: ObservationFactory = ObservationBuilder,
    ) -> None:
        super().__init__()
        if role not in {"train", "validation", "test"}:
            raise ValueError("role must be train, validation, or test")
        if isinstance(worker_index, bool) or not isinstance(worker_index, int):
            raise ValueError("worker_index must be a non-negative integer")
        split = getattr(config.scenarios, role)
        self._seed_allocator = EpisodeSeedAllocator(role, split, worker_index)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(24,),
            dtype=np.float32,
        )
        self._config = config
        self._role = role
        self._worker_index = worker_index
        self._scenario_runtime_factory = scenario_runtime_factory
        self._env_factory = env_factory
        self._suite_factory = suite_factory
        self._shield_factory = shield_factory
        self._builder_factory = builder_factory
        self._reward_factory = reward_factory
        self._observation_factory = observation_factory
        self._environment: DrivingEnvironment | None = None
        self._suite: AnalysisSuite | None = None
        self._shield: Shield | None = None
        self._builder: FrameBuilder | None = None
        self._reward: Reward | None = None
        self._observation: Observation | None = None
        self._runtime: ScenarioRuntime | None = None
        self._scenario_state: ScenarioState | None = None
        self._frame: SceneFrame | None = None
        self._analysis: AgentAnalysisResult | None = None
        self._episode_seeds: EpisodeSeeds | None = None
        self._actual_scenario_index: int | None = None
        self._episode_active = False
        self._gym_rng_initialized = False
        self._closed = False
        self._fatally_closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        """Reset every episode-owned component and return the initial observation."""

        self._require_resettable()
        del options
        episode_rng_seed = self._next_episode_rng_seed(seed)
        seeds = self._seed_allocator.allocate(episode_rng_seed)
        environment = self._environment
        self._clear_episode()
        try:
            if environment is None:
                environment = self._new_environment()
            self._environment = environment

            suite = self._suite_factory(self._config.agents)
            shield = self._shield_factory(self._config.shield)
            builder = self._builder_factory()
            reward = self._reward_factory(self._config.reward)
            reward.reset()
            observation_builder = self._observation_factory(self._config.observation)
            runtime = self._scenario_runtime_factory(self._config.scenario_id)
            self._validate_runtime(runtime)

            scenario_state = runtime.reset(environment, seeds=seeds)
            if not isinstance(scenario_state, ScenarioState):
                raise TypeError("ScenarioRuntime.reset must return ScenarioState")
            _, raw_reset_info = environment.reset(seed=seeds.metadrive_scenario_index)
            reset_info = self._copy_info(raw_reset_info)
            actual_scenario_index = self._verified_actual_scenario_index(
                environment,
                reset_info,
                seeds.metadrive_scenario_index,
            )
            runtime.after_simulator_reset(environment, scenario_state)
            context = self._runtime_context(runtime, scenario_state)
            initial_result = ScenarioStepResult(success=False, failure=False)
            frame = self._build_frame(
                builder,
                environment,
                step_index=0,
                seeds=seeds,
                context=context,
                scenario_result=initial_result,
                raw_info=reset_info,
                previous_executed_action=int(DrivingAction.KEEP),
                previous_shield_intervention=False,
            )
            analysis = self._analyze(suite, frame.observation)
            observation = self._build_observation(
                observation_builder,
                frame.observation,
                analysis,
            )
            info = reset_info
            info.update(self._episode_metadata(seeds, actual_scenario_index))
        except Exception as error:
            self._fatal_close(error)
            raise

        self._suite = suite
        self._shield = shield
        self._builder = builder
        self._reward = reward
        self._observation = observation_builder
        self._runtime = runtime
        self._scenario_state = scenario_state
        self._frame = frame
        self._analysis = analysis
        self._episode_seeds = seeds
        self._actual_scenario_index = actual_scenario_index
        self._episode_active = True
        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        """Filter one requested action, advance the simulator, and analyze the transition."""

        self._require_active_episode()
        requested = self._validated_action(action)
        environment = cast(DrivingEnvironment, self._environment)
        suite = cast(AnalysisSuite, self._suite)
        shield = cast(Shield, self._shield)
        builder = cast(FrameBuilder, self._builder)
        reward_calculator = cast(Reward, self._reward)
        observation_builder = cast(Observation, self._observation)
        runtime = cast(ScenarioRuntime, self._runtime)
        scenario_state = cast(ScenarioState, self._scenario_state)
        frame = cast(SceneFrame, self._frame)
        analysis = cast(AgentAnalysisResult, self._analysis)
        seeds = cast(EpisodeSeeds, self._episode_seeds)
        actual_scenario_index = cast(int, self._actual_scenario_index)

        try:
            shield_result = shield.filter(
                requested,
                frame.observation,
                analysis.claims,
            )
            executed = shield_result.executed_action
            target = target_speed_mps(
                executed,
                frame.observation.ego.speed_mps,
                frame.observation.ego.speed_limit_mps,
            )
            step_index = frame.observation.step_index + 1
            runtime.before_step(environment, scenario_state, step_index=step_index)
            _, _, raw_terminated, raw_truncated, raw_step_info = environment.step(int(executed))
            step_info = self._copy_info(raw_step_info)
            scenario_result = runtime.after_step(
                environment,
                scenario_state,
                step_index=step_index,
                raw_info=step_info,
            )
            if not isinstance(scenario_result, ScenarioStepResult):
                raise TypeError("ScenarioRuntime.after_step must return ScenarioStepResult")
            context = self._runtime_context(runtime, scenario_state)
            next_frame = self._build_frame(
                builder,
                environment,
                step_index=step_index,
                seeds=seeds,
                context=context,
                scenario_result=scenario_result,
                raw_info=step_info,
                previous_executed_action=int(executed),
                previous_shield_intervention=shield_result.intervened,
            )
            next_analysis = self._analyze(suite, next_frame.observation)
            reward_result = reward_calculator.calculate(
                RewardContext(
                    previous_frame=frame,
                    next_frame=next_frame,
                    analysis=next_analysis,
                    executed_action=int(executed),
                    shield_intervened=shield_result.intervened,
                    decision_interval_s=decision_interval_s(environment.config),
                )
            )
            observation = self._build_observation(
                observation_builder,
                next_frame.observation,
                next_analysis,
            )
            privileged = next_frame.privileged
            terminated = bool(raw_terminated) or any(
                (
                    privileged.collision_occurred,
                    privileged.arrived,
                    privileged.scenario_success,
                    privileged.scenario_failure,
                )
            )
            truncated = bool(raw_truncated)
            trace = DecisionTrace(
                step_index=step_index,
                raw_action=int(requested),
                executed_action=int(executed),
                target_speed_mps=target,
                shield_intervened=shield_result.intervened,
                shield_reasons=shield_result.reasons,
                claims=analysis.claims,
                review=analysis.review,
                reward_components=reward_result.components,
                episode_rng_seed=seeds.episode_rng_seed,
                metadrive_scenario_index=actual_scenario_index,
                scenario_parameter_seed=seeds.scenario_parameter_seed,
                role=self._role,
                worker_index=self._worker_index,
            )
            info = step_info
            info.update(self._episode_metadata(seeds, actual_scenario_index))
            info.update(
                {
                    "requested_action": int(requested),
                    "executed_action": int(executed),
                    "shield_intervened": shield_result.intervened,
                    "shield_reasons": tuple(shield_result.reasons),
                    "target_speed_mps": target,
                    "failed_agent_ids": tuple(next_analysis.failed_agent_ids),
                    "analysis_errors": tuple(next_analysis.errors),
                    "reward_components": dict(reward_result.components),
                    "decision_trace": trace,
                }
            )
            reward_total = float(reward_result.total)
        except Exception as error:
            self._fatal_close(error)
            raise

        self._frame = next_frame
        self._analysis = next_analysis
        self._episode_active = not (terminated or truncated)
        return observation, reward_total, terminated, truncated, info

    def close(self) -> None:
        """Close the owned simulator exactly once and make this wrapper terminal."""

        if self._closed:
            return
        self._closed = True
        environment = self._detach_environment()
        if environment is None:
            return
        try:
            environment.close()
        except Exception:
            pass

    def _new_environment(self) -> DrivingEnvironment:
        metadrive_config = self._config.metadrive_dict()
        split = getattr(self._config.scenarios, self._role)
        metadrive_config.update(
            {
                "start_seed": split.seed_start,
                "num_scenarios": split.seed_count,
            }
        )
        return self._env_factory(metadrive_config, self._config.control)

    def _next_episode_rng_seed(self, requested_seed: int | None) -> int:
        if requested_seed is not None:
            super().reset(seed=requested_seed)
            self._gym_rng_initialized = True
            return requested_seed
        if not self._gym_rng_initialized:
            super().reset(seed=self._config.seed)
            self._gym_rng_initialized = True
            return self._config.seed
        super().reset(seed=None)
        return int(self.np_random.integers(0, np.iinfo(np.int32).max))

    def _fatal_close(self, primary_error: Exception) -> None:
        self._fatally_closed = True
        environment = self._detach_environment()
        if environment is None:
            return
        try:
            environment.close()
        except Exception as cleanup_error:
            primary_error.add_note(f"simulator cleanup failed: {cleanup_error}")

    def _detach_environment(self) -> DrivingEnvironment | None:
        environment = self._environment
        self._environment = None
        self._clear_episode()
        return environment

    def _clear_episode(self) -> None:
        self._suite = None
        self._shield = None
        self._builder = None
        self._reward = None
        self._observation = None
        self._runtime = None
        self._scenario_state = None
        self._frame = None
        self._analysis = None
        self._episode_seeds = None
        self._actual_scenario_index = None
        self._episode_active = False

    def _require_resettable(self) -> None:
        if self._fatally_closed:
            raise RuntimeError("environment is fatally closed after an internal error")
        if self._closed:
            raise RuntimeError("environment is closed")

    def _require_active_episode(self) -> None:
        if not self._episode_active:
            raise RuntimeError("reset() must be called before step()")

    def _validated_action(self, action: int) -> DrivingAction:
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise ValueError("action must be an integer from 0 through 3")
        value = int(action)
        if not self.action_space.contains(value):
            raise ValueError("action must be an integer from 0 through 3")
        return DrivingAction(value)

    def _episode_metadata(
        self,
        seeds: EpisodeSeeds,
        actual_scenario_index: int,
    ) -> dict[str, object]:
        return {
            "episode_rng_seed": int(seeds.episode_rng_seed),
            "simulator_seed": int(seeds.metadrive_scenario_index),
            "scenario_seed": int(seeds.scenario_parameter_seed),
            "metadrive_scenario_index": int(actual_scenario_index),
            "scenario_parameter_seed": int(seeds.scenario_parameter_seed),
            "role": self._role,
            "worker_index": int(self._worker_index),
        }

    @staticmethod
    def _copy_info(raw_info: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(raw_info, Mapping):
            raise TypeError("simulator info must be a mapping")
        return dict(raw_info)

    @staticmethod
    def _verified_actual_scenario_index(
        environment: DrivingEnvironment,
        raw_info: Mapping[str, object],
        requested_index: int,
    ) -> int:
        reported = raw_info.get("env_seed")
        current = getattr(environment, "current_seed", None)
        candidates = (reported, current)
        if all(candidate is None for candidate in candidates):
            raise RuntimeError("simulator did not report its actual scenario index")
        actual_values: list[int] = []
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, bool) or not isinstance(candidate, Integral):
                raise TypeError("simulator scenario index must be an integer")
            actual_values.append(int(candidate))
        if any(actual != requested_index for actual in actual_values):
            raise RuntimeError(
                "simulator scenario index mismatch: "
                f"requested {requested_index}, returned {actual_values}"
            )
        return actual_values[0]

    @staticmethod
    def _validate_runtime(runtime: object) -> None:
        methods = (
            "reset",
            "after_simulator_reset",
            "before_step",
            "after_step",
            "observation_context",
        )
        if not all(callable(getattr(runtime, method, None)) for method in methods):
            raise TypeError("scenario_runtime_factory must return a ScenarioRuntime")

    @staticmethod
    def _runtime_context(
        runtime: ScenarioRuntime,
        state: ScenarioState,
    ) -> ScenarioObservationContext:
        context = runtime.observation_context(state)
        if not isinstance(context, ScenarioObservationContext):
            raise TypeError(
                "ScenarioRuntime.observation_context must return ScenarioObservationContext"
            )
        return context

    @staticmethod
    def _build_frame(
        builder: FrameBuilder,
        environment: DrivingEnvironment,
        **kwargs: Any,
    ) -> SceneFrame:
        frame = builder.build(environment, **kwargs)
        if not isinstance(frame, SceneFrame):
            raise TypeError("frame builder must return SceneFrame")
        return frame

    @staticmethod
    def _analyze(
        suite: AnalysisSuite,
        observation: SceneObservation,
    ) -> AgentAnalysisResult:
        analysis = analyze_safely(suite, observation)
        if not isinstance(analysis, AgentAnalysisResult):
            raise TypeError("analyze_safely must return AgentAnalysisResult")
        return analysis

    def _build_observation(
        self,
        observation_builder: Observation,
        scene_observation: SceneObservation,
        analysis: AgentAnalysisResult,
    ) -> NDArray[np.float32]:
        observation = observation_builder.build(
            scene_observation,
            analysis.claims,
            analysis.review,
        )
        if not self.observation_space.contains(observation):
            raise ValueError("observation is outside the declared observation space")
        return observation


def create_metadrive_env(config: dict[str, object]) -> DrivingEnvironment:
    """Construct MetaDrive lazily so unit tests stay simulator-independent."""

    from metadrive import MetaDriveEnv  # type: ignore[import-untyped]

    return cast(DrivingEnvironment, MetaDriveEnv(config))
