"""Scenario selection and lifecycle delegation at reset boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from numbers import Integral
from typing import TYPE_CHECKING

from mad_driving.config.models import ScenarioSplitsConfig
from mad_driving.scenarios.cut_in import CutInRuntime
from mad_driving.scenarios.lead_brake import LeadBrakeRuntime, NominalScenarioRuntime
from mad_driving.scenarios.occluded_crossing import OccludedCrossingRuntime
from mad_driving.scenarios.parameters import ScenarioParameterSampler
from mad_driving.scenarios.runtime import (
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

ScenarioRuntimeBuilder = Callable[[int, ScenarioParameterSampler], ScenarioRuntime]


class ScenarioManagerRuntime:
    """Select deterministic concrete scenarios and delegate their lifecycle."""

    def __init__(
        self,
        config: ScenarioSplitsConfig,
        *,
        runtimes: Mapping[str, ScenarioRuntimeBuilder] | None = None,
    ) -> None:
        self._config = config
        self._pending_level = self._initial_level(config)
        self._scenario_schedule: list[str] = []
        self._active_runtime: ScenarioRuntime | None = None
        self._runtimes = self._default_runtimes() if runtimes is None else dict(runtimes)

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

    def set_scenario_schedule(self, scenario_ids: Sequence[str]) -> None:
        """Replace the finite reset schedule used by deterministic validation."""

        if isinstance(scenario_ids, str | bytes) or not isinstance(scenario_ids, Sequence):
            raise TypeError("scenario schedule must be a sequence of scenario IDs")
        schedule = tuple(scenario_ids)
        if not schedule or not all(isinstance(item, str) and item for item in schedule):
            raise ValueError("scenario schedule must contain non-empty scenario IDs")
        allowed = frozenset(_SCENARIOS_BY_LEVEL[self._pending_level])
        outside_level = set(schedule) - allowed
        if outside_level:
            raise ValueError(
                "scenario schedule contains scenarios outside the pending difficulty level: "
                f"{sorted(outside_level)!r}"
            )
        self._scenario_schedule = list(schedule)

    def reset(self, environment: DrivingEnvironment, *, seeds: EpisodeSeeds) -> ScenarioState:
        """Select, create, and initialize one concrete scenario runtime."""

        level = self._pending_level
        selection_sampler = ScenarioParameterSampler(seeds.scenario_selection_seed)
        parameter_sampler = ScenarioParameterSampler(seeds.scenario_parameter_seed)
        scenario_id = (
            self._scenario_schedule.pop(0)
            if self._scenario_schedule
            else self._select_scenario(level, selection_sampler)
        )
        runtime = self._create_runtime(scenario_id, level, parameter_sampler)
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
        builder = self._runtimes.get(scenario_id)
        if builder is None:
            raise RuntimeError(f"no registered runtime for selected scenario: {scenario_id}")
        return builder(level, sampler)

    def _select_scenario(self, level: int, sampler: ScenarioParameterSampler) -> str:
        allowed = _SCENARIOS_BY_LEVEL[level]
        selection = self._config.selection
        if selection == "auto":
            return sampler.choose(allowed)
        if selection not in allowed:
            raise ValueError(
                f"scenario selection {selection!r} is not allowed at difficulty level {level}"
            )
        return selection

    def _default_runtimes(self) -> dict[str, ScenarioRuntimeBuilder]:
        return {
            "nominal": lambda level, sampler: NominalScenarioRuntime(
                survival_s=self._config.lead_brake.survival_s
            ),
            "lead_brake": lambda level, sampler: LeadBrakeRuntime(
                self._config.lead_brake,
                sampler,
                difficulty_level=level,
            ),
            "cut_in": lambda level, sampler: CutInRuntime(
                self._config.cut_in,
                sampler,
                difficulty_level=level,
            ),
            "occluded_crossing": lambda level, sampler: OccludedCrossingRuntime(
                self._config.occluded_crossing,
                sampler,
                difficulty_level=level,
            ),
        }
