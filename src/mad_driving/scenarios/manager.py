"""Scenario selection and lifecycle delegation at reset boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from typing import TYPE_CHECKING

from mad_driving.config.models import ScenarioSplitsConfig
from mad_driving.scenarios.lead_brake import LeadBrakeRuntime, NominalScenarioRuntime
from mad_driving.scenarios.parameters import ScenarioParameterSampler
from mad_driving.scenarios.runtime import (
    NoOpScenarioRuntime,
    ScenarioObservationContext,
    ScenarioRuntime,
    ScenarioState,
    ScenarioTransition,
)
from mad_driving.scenarios.seeding import EpisodeSeeds

if TYPE_CHECKING:
    from mad_driving.envs.multi_agent_speed_env import DrivingEnvironment


_SCENARIOS_BY_LEVEL: dict[int, tuple[str, ...]] = {
    0: ("nominal",),
    1: ("lead_brake",),
    2: ("lead_brake", "cut_in"),
    3: ("occluded_crossing",),
}


class ScenarioManagerRuntime:
    """Select deterministic concrete scenarios and delegate their lifecycle."""

    def __init__(self, config: ScenarioSplitsConfig) -> None:
        self._config = config
        self._pending_level = self._initial_level(config)
        self._active_runtime: ScenarioRuntime | None = None

    @staticmethod
    def _initial_level(config: ScenarioSplitsConfig) -> int:
        if config.curriculum.mode == "fixed":
            return config.curriculum.fixed_level
        return config.curriculum.initial_level

    def set_difficulty_level(self, level: int) -> None:
        """Queue a validated level for the following reset only."""

        if (
            isinstance(level, bool)
            or not isinstance(level, Integral)
            or level not in _SCENARIOS_BY_LEVEL
        ):
            raise ValueError("difficulty level must be an integer from 0 through 3")
        self._pending_level = int(level)

    def reset(self, environment: DrivingEnvironment, *, seeds: EpisodeSeeds) -> ScenarioState:
        """Select, create, and initialize one concrete scenario runtime."""

        level = self._pending_level
        sampler = ScenarioParameterSampler(seeds.scenario_parameter_seed)
        scenario_id = sampler.choose(_SCENARIOS_BY_LEVEL[level])
        runtime = self._create_runtime(scenario_id, level, sampler)
        state = runtime.reset(environment, seeds=seeds)
        parameters = dict(state.parameters)
        parameters["difficulty_level"] = level
        self._active_runtime = runtime
        return replace(state, parameters=parameters)

    def after_simulator_reset(
        self, environment: DrivingEnvironment, state: ScenarioState
    ) -> ScenarioState:
        return self._runtime().after_simulator_reset(environment, state)

    def before_step(
        self, environment: DrivingEnvironment, state: ScenarioState, *, step_index: int
    ) -> ScenarioState:
        return self._runtime().before_step(environment, state, step_index=step_index)

    def after_step(
        self,
        environment: DrivingEnvironment,
        state: ScenarioState,
        *,
        step_index: int,
        raw_info: Mapping[str, object],
    ) -> ScenarioTransition:
        return self._runtime().after_step(
            environment, state, step_index=step_index, raw_info=raw_info
        )

    def observation_context(self, state: ScenarioState) -> ScenarioObservationContext:
        return self._runtime().observation_context(state)

    def _runtime(self) -> ScenarioRuntime:
        if self._active_runtime is None:
            raise RuntimeError("ScenarioManagerRuntime.reset must be called before lifecycle hooks")
        return self._active_runtime

    def _create_runtime(
        self,
        scenario_id: str,
        level: int,
        sampler: ScenarioParameterSampler,
    ) -> ScenarioRuntime:
        if scenario_id == "nominal":
            return NominalScenarioRuntime(survival_s=self._config.lead_brake.survival_s)
        if scenario_id == "lead_brake":
            return LeadBrakeRuntime(self._config.lead_brake, sampler, difficulty_level=level)
        return NoOpScenarioRuntime(scenario_id)
