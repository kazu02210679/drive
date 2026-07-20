"""Simulator lifecycle boundaries and the concrete Gymnasium environment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from mad_driving.agents.suite import (
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
from mad_driving.envs.reward import (
    CollisionKind,
    RewardCalculator,
    RewardContext,
    RewardResult,
)
from mad_driving.interfaces import (
    CriticReview,
    DecisionTrace,
    RiskClaim,
    SceneSnapshot,
    ShieldResult,
)
from mad_driving.safety import SafetyShield
from mad_driving.world_model import SceneSnapshotBuilder
from mad_driving.world_model.validation import decision_interval_s


class DrivingEnvironment(Protocol):
    """Small subset of the MetaDrive environment API required by Phase 1."""

    config: dict[str, Any]
    vehicle: Any
    engine: Any
    agent: Any
    action_space: Any

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
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
    ) -> ShieldResult: ...


class ShieldFactory(Protocol):
    def __call__(self, config: ShieldConfig) -> Shield: ...


class SnapshotBuilder(Protocol):
    def build(
        self,
        env: DrivingEnvironment,
        *,
        step_index: int,
        scenario_id: str,
        seed: int,
        previous_action: int,
        previous_shield_intervention: bool,
    ) -> SceneSnapshot: ...


class SnapshotBuilderFactory(Protocol):
    def __call__(self) -> SnapshotBuilder: ...


class Reward(Protocol):
    def reset(self) -> None: ...

    def calculate(self, context: RewardContext) -> RewardResult: ...


class RewardFactory(Protocol):
    def __call__(self, config: RewardConfig) -> Reward: ...


class Observation(Protocol):
    def build(
        self,
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> NDArray[np.float32]: ...


class ObservationFactory(Protocol):
    def __call__(self, config: ObservationConfig) -> Observation: ...


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
        env_factory: ControlEnvironmentFactory = _create_control_environment,
        suite_factory: SuiteFactory = AgentSuite.from_config,
        shield_factory: ShieldFactory = SafetyShield,
        builder_factory: SnapshotBuilderFactory = SceneSnapshotBuilder,
        reward_factory: RewardFactory = RewardCalculator,
        observation_factory: ObservationFactory = ObservationBuilder,
    ) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(24,),
            dtype=np.float32,
        )
        self._config = config
        self._env_factory = env_factory
        self._suite_factory = suite_factory
        self._shield_factory = shield_factory
        self._builder_factory = builder_factory
        self._reward_factory = reward_factory
        self._observation_factory = observation_factory
        self._environment: DrivingEnvironment | None = None
        self._suite: AnalysisSuite | None = None
        self._shield: Shield | None = None
        self._builder: SnapshotBuilder | None = None
        self._reward: Reward | None = None
        self._observation: Observation | None = None
        self._snapshot: SceneSnapshot | None = None
        self._claims: tuple[RiskClaim, ...] | None = None
        self._review: CriticReview | None = None
        self._episode_seed: int | None = None
        self._episode_active = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        """Reset simulator-owned state and return the initial fixed observation."""

        environment = self._environment
        self._clear_episode()
        episode_seed = self._config.seed if seed is None else seed
        try:
            super().reset(seed=seed)
            del options
            if environment is None:
                environment = self._new_environment()
            self._environment = environment

            suite = self._suite_factory(self._config.agents)
            shield = self._shield_factory(self._config.shield)
            builder = self._builder_factory()
            reward = self._reward_factory(self._config.reward)
            reward.reset()
            observation_builder = self._observation_factory(self._config.observation)

            _, raw_info = environment.reset(seed=episode_seed)
            snapshot = builder.build(
                environment,
                step_index=0,
                scenario_id=self._config.scenario_id,
                seed=episode_seed,
                previous_action=int(DrivingAction.KEEP),
                previous_shield_intervention=False,
            )
            claims, review = analyze_safely(suite, snapshot)
            observation = self._build_observation(
                observation_builder,
                snapshot,
                claims,
                review,
            )
            info: dict[str, object] = dict(raw_info)
            info["seed"] = episode_seed
        except Exception:
            self._environment = environment
            self._close_simulator()
            raise

        self._suite = suite
        self._shield = shield
        self._builder = builder
        self._reward = reward
        self._observation = observation_builder
        self._snapshot = snapshot
        self._claims = claims
        self._review = review
        self._episode_seed = episode_seed
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
        builder = cast(SnapshotBuilder, self._builder)
        reward_calculator = cast(Reward, self._reward)
        observation_builder = cast(Observation, self._observation)
        snapshot = cast(SceneSnapshot, self._snapshot)
        claims = cast(tuple[RiskClaim, ...], self._claims)
        review = cast(CriticReview, self._review)
        episode_seed = cast(int, self._episode_seed)

        shield_result = shield.filter(requested, snapshot, claims)
        executed = shield_result.executed_action
        target = target_speed_mps(
            executed,
            snapshot.ego.speed_mps,
            snapshot.ego.speed_limit_mps,
        )

        try:
            _, _, raw_terminated, raw_truncated, raw_info = environment.step(int(executed))
        except Exception as exc:
            self._close_simulator()
            return (
                self._safe_observation(),
                0.0,
                False,
                True,
                {
                    "requested_action": int(requested),
                    "executed_action": int(executed),
                    "simulator_error": str(exc),
                },
            )

        step_index = snapshot.step_index + 1
        try:
            next_snapshot = builder.build(
                environment,
                step_index=step_index,
                scenario_id=self._config.scenario_id,
                seed=episode_seed,
                previous_action=int(executed),
                previous_shield_intervention=shield_result.intervened,
            )
        except Exception as exc:
            self._episode_active = False
            return (
                self._safe_observation(),
                0.0,
                False,
                True,
                {"analysis_error": str(exc)},
            )

        next_claims, next_review = analyze_safely(suite, next_snapshot)
        reward_result = reward_calculator.calculate(
            RewardContext(
                previous_snapshot=snapshot,
                next_snapshot=next_snapshot,
                post_step_claims=next_claims,
                executed_action=int(executed),
                shield_intervened=shield_result.intervened,
                arrived=bool(raw_info.get("arrive_dest", False)),
                collision_kind=self._collision_kind(raw_info, next_snapshot),
                decision_interval_s=decision_interval_s(environment.config),
            )
        )
        observation, observation_error = self._build_observation_safely(
            observation_builder,
            next_snapshot,
            next_claims,
            next_review,
        )
        terminated = bool(raw_terminated)
        truncated = bool(raw_truncated) or observation_error is not None
        trace = DecisionTrace(
            step_index=step_index,
            raw_action=int(requested),
            executed_action=int(executed),
            target_speed_mps=target,
            shield_intervened=shield_result.intervened,
            shield_reasons=shield_result.reasons,
            claims=claims,
            review=review,
            reward_components=reward_result.components,
        )
        info = dict(raw_info)
        info.update(
            {
                "requested_action": int(requested),
                "executed_action": int(executed),
                "shield_intervened": shield_result.intervened,
                "shield_reasons": shield_result.reasons,
                "target_speed_mps": target,
                "reward_components": dict(reward_result.components),
                "decision_trace": trace,
            }
        )
        if observation_error is not None:
            info["observation_error"] = observation_error

        self._snapshot = next_snapshot
        self._claims = next_claims
        self._review = next_review
        self._episode_active = not (terminated or truncated)
        return observation, reward_result.total, terminated, truncated, info

    def close(self) -> None:
        """Close the current simulator instance at most once."""

        self._close_simulator()

    def _new_environment(self) -> DrivingEnvironment:
        return self._env_factory(self._config.metadrive_dict(), self._config.control)

    def _close_simulator(self) -> None:
        environment = self._environment
        self._environment = None
        self._clear_episode()
        if environment is None:
            return
        try:
            environment.close()
        except Exception:
            pass

    def _clear_episode(self) -> None:
        self._suite = None
        self._shield = None
        self._builder = None
        self._reward = None
        self._observation = None
        self._snapshot = None
        self._claims = None
        self._review = None
        self._episode_seed = None
        self._episode_active = False

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

    def _build_observation_safely(
        self,
        observation_builder: Observation,
        snapshot: SceneSnapshot,
        claims: tuple[RiskClaim, ...],
        review: CriticReview,
    ) -> tuple[NDArray[np.float32], str | None]:
        try:
            return self._build_observation(observation_builder, snapshot, claims, review), None
        except Exception as exc:
            return self._safe_observation(), str(exc)

    def _build_observation(
        self,
        observation_builder: Observation,
        snapshot: SceneSnapshot,
        claims: tuple[RiskClaim, ...],
        review: CriticReview,
    ) -> NDArray[np.float32]:
        observation = observation_builder.build(snapshot, claims, review)
        if not self.observation_space.contains(observation):
            raise ValueError("observation is outside the declared observation space")
        return observation

    def _safe_observation(self) -> NDArray[np.float32]:
        return np.zeros((24,), dtype=np.float32)

    @staticmethod
    def _collision_kind(
        raw_info: dict[str, Any],
        next_snapshot: SceneSnapshot,
    ) -> CollisionKind | None:
        if bool(raw_info.get("crash_human", False)):
            return "crossing_actor"
        if bool(raw_info.get("crash_vehicle", False)) or next_snapshot.collision_occurred:
            return "vehicle"
        return None


def create_metadrive_env(config: dict[str, object]) -> DrivingEnvironment:
    """Construct MetaDrive lazily so unit tests stay simulator-independent."""

    from metadrive import MetaDriveEnv  # type: ignore[import-untyped]

    return cast(DrivingEnvironment, MetaDriveEnv(config))
