"""Cut-in scenario trajectory helpers and runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import ceil
from typing import TYPE_CHECKING

from mad_driving.config.models import CutInScenarioConfig
from mad_driving.scenarios.actors import LanePoseCommand, LaneVehicleSpawn
from mad_driving.scenarios.parameters import ScenarioParameterSampler
from mad_driving.scenarios.runtime import (
    ScenarioObservationContext,
    ScenarioState,
    ScenarioStepResult,
    ScenarioTransition,
)
from mad_driving.scenarios.seeding import EpisodeSeeds

if TYPE_CHECKING:
    from mad_driving.envs.multi_agent_speed_env import DrivingEnvironment


_CUT_IN_ACTOR_ID = "cut-in"


def smoothstep(progress: float) -> float:
    """Return a bounded cubic interpolation weight."""

    bounded = min(max(progress, 0.0), 1.0)
    return bounded * bounded * (3.0 - 2.0 * bounded)


class CutInRuntime:
    """Spawn a lead vehicle in an adjacent lane and merge it into the ego lane."""

    def __init__(
        self,
        config: CutInScenarioConfig,
        sampler: ScenarioParameterSampler,
        *,
        difficulty_level: int,
    ) -> None:
        self._config = config
        self._sampler = sampler
        self._difficulty_level = difficulty_level

    def reset(self, environment: DrivingEnvironment, *, seeds: EpisodeSeeds) -> ScenarioState:
        del environment
        return ScenarioState(
            "cut_in",
            seeds,
            {
                "difficulty_level": self._difficulty_level,
                "initial_gap_m": self._sampler.uniform(
                    "initial_gap_m",
                    self._config.initial_gap_m.minimum,
                    self._config.initial_gap_m.maximum,
                ),
                "trigger_s": self._sampler.uniform(
                    "trigger_s", self._config.trigger_s.minimum, self._config.trigger_s.maximum
                ),
                "merge_duration_s": self._sampler.uniform(
                    "merge_duration_s",
                    self._config.merge_duration_s.minimum,
                    self._config.merge_duration_s.maximum,
                ),
                "speed_fraction": self._sampler.uniform(
                    "speed_fraction",
                    self._config.speed_fraction.minimum,
                    self._config.speed_fraction.maximum,
                ),
                "survival_s": self._config.survival_s,
                "actor_id": _CUT_IN_ACTOR_ID,
            },
        )

    def after_simulator_reset(
        self, environment: DrivingEnvironment, state: ScenarioState
    ) -> ScenarioState:
        geometry = environment.scenario_road_geometry()
        if not geometry.adjacent_lane_indices:
            raise RuntimeError("cut-in requires an available adjacent lane")
        source_lane_index = self._sampler.choose(geometry.adjacent_lane_indices)
        source_lateral_m = (
            source_lane_index[2] - geometry.ego_lane_index[2]
        ) * geometry.lane_width_m
        initial_longitudinal_m = (
            geometry.ego_longitudinal_m + self._parameter_float(state, "initial_gap_m")
        )
        speed_mps = geometry.ego_speed_mps * self._parameter_float(state, "speed_fraction")
        actor_id = self._parameter_str(state, "actor_id")
        spawned_id = environment.scenario_spawn_lane_vehicle(
            LaneVehicleSpawn(actor_id, source_lane_index, initial_longitudinal_m, 0.0, speed_mps)
        )
        if spawned_id != actor_id:
            raise RuntimeError("scenario Actor spawn returned an unexpected ID")
        trigger_step = ceil(
            self._parameter_float(state, "trigger_s") / geometry.decision_interval_s
        )
        merge_steps = ceil(
            self._parameter_float(state, "merge_duration_s") / geometry.decision_interval_s
        )
        success_step = trigger_step + merge_steps + ceil(
            self._parameter_float(state, "survival_s") / geometry.decision_interval_s
        )
        return replace(
            state,
            parameters={
                **state.parameters,
                "source_lane_index": source_lane_index,
                "source_lateral_m": source_lateral_m,
                "initial_longitudinal_m": initial_longitudinal_m,
                "speed_mps": speed_mps,
                "ego_lane_index": geometry.ego_lane_index,
                "trigger_step": trigger_step,
                "merge_steps": merge_steps,
                "success_step": success_step,
                "decision_interval_s": geometry.decision_interval_s,
            },
        )

    def before_step(
        self, environment: DrivingEnvironment, state: ScenarioState, *, step_index: int
    ) -> ScenarioState:
        actor_id = self._require_spawned_actor(environment, state)
        trigger_step = self._parameter_int(state, "trigger_step")
        if step_index >= trigger_step:
            merge_steps = self._parameter_int(state, "merge_steps")
            progress = (step_index - trigger_step) / merge_steps
            lateral_m = self._parameter_float(state, "source_lateral_m") * (
                1.0 - smoothstep(progress)
            )
            longitudinal_m = self._parameter_float(state, "initial_longitudinal_m") + (
                self._parameter_float(state, "speed_mps")
                * self._parameter_float(state, "decision_interval_s")
                * step_index
            )
            environment.scenario_command_actor(
                actor_id,
                LanePoseCommand(
                    self._parameter_lane_index(state, "ego_lane_index"), longitudinal_m, lateral_m
                ),
            )
        return state

    def after_step(
        self,
        environment: DrivingEnvironment,
        state: ScenarioState,
        *,
        step_index: int,
        raw_info: Mapping[str, object],
    ) -> ScenarioTransition:
        self._require_spawned_actor(environment, state)
        collision = bool(raw_info.get("crash_vehicle", False))
        success = not collision and step_index >= self._parameter_int(state, "success_step")
        return ScenarioTransition(state, ScenarioStepResult(success=success, failure=collision))

    def observation_context(self, state: ScenarioState) -> ScenarioObservationContext:
        return ScenarioObservationContext(scenario_id=state.scenario_id)

    @staticmethod
    def _parameter_float(state: ScenarioState, name: str) -> float:
        value = state.parameters[name]
        if isinstance(value, bool) or not isinstance(value, float):
            raise RuntimeError(f"scenario parameter {name} must be a float")
        return value

    @staticmethod
    def _parameter_int(state: ScenarioState, name: str) -> int:
        value = state.parameters[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"scenario parameter {name} must be an integer")
        return value

    @staticmethod
    def _parameter_str(state: ScenarioState, name: str) -> str:
        value = state.parameters[name]
        if not isinstance(value, str):
            raise RuntimeError(f"scenario parameter {name} must be a string")
        return value

    @staticmethod
    def _parameter_lane_index(state: ScenarioState, name: str) -> tuple[str, str, int]:
        value = state.parameters[name]
        if (
            not isinstance(value, tuple)
            or len(value) != 3
            or not isinstance(value[0], str)
            or not isinstance(value[1], str)
            or isinstance(value[2], bool)
            or not isinstance(value[2], int)
        ):
            raise RuntimeError(f"scenario parameter {name} must be a lane index")
        return value

    def _require_spawned_actor(self, environment: DrivingEnvironment, state: ScenarioState) -> str:
        actor_id = self._parameter_str(state, "actor_id")
        if actor_id not in environment.scenario_actor_ids():
            raise RuntimeError(f"missing scenario Actor: {actor_id}")
        return actor_id
