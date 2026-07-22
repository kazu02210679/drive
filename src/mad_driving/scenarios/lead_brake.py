"""Nominal and Lead Brake scenario runtime implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import ceil
from typing import TYPE_CHECKING

from mad_driving.config.models import LeadBrakeScenarioConfig
from mad_driving.scenarios.actors import ActorCommand, LaneVehicleSpawn
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


_LEAD_ACTOR_ID = "lead-brake"


class NominalScenarioRuntime:
    """Level-zero runtime with a meaningful collision-free survival success."""

    def __init__(self, *, survival_s: float) -> None:
        self._survival_s = survival_s
        self._success_step: int | None = None

    def reset(self, environment: DrivingEnvironment, *, seeds: EpisodeSeeds) -> ScenarioState:
        del environment
        self._success_step = None
        return ScenarioState("nominal", seeds, {"survival_s": self._survival_s})

    def after_simulator_reset(
        self, environment: DrivingEnvironment, state: ScenarioState
    ) -> ScenarioState:
        geometry = environment.scenario_road_geometry()
        self._success_step = ceil(self._survival_s / geometry.decision_interval_s)
        return replace(state, parameters={**state.parameters, "success_step": self._success_step})

    def before_step(
        self, environment: DrivingEnvironment, state: ScenarioState, *, step_index: int
    ) -> ScenarioState:
        del environment, step_index
        return state

    def after_step(
        self,
        environment: DrivingEnvironment,
        state: ScenarioState,
        *,
        step_index: int,
        raw_info: Mapping[str, object],
    ) -> ScenarioTransition:
        del environment
        collision = bool(raw_info.get("crash_vehicle", False))
        off_road = bool(raw_info.get("out_of_road", False))
        success = not collision and not off_road and step_index >= self._require_success_step()
        return ScenarioTransition(
            state,
            ScenarioStepResult(success=success, failure=collision or off_road),
        )

    def observation_context(self, state: ScenarioState) -> ScenarioObservationContext:
        return ScenarioObservationContext(scenario_id=state.scenario_id)

    def _require_success_step(self) -> int:
        if self._success_step is None:
            raise RuntimeError("NominalScenarioRuntime.after_simulator_reset must be called first")
        return self._success_step


class LeadBrakeRuntime:
    """Spawn a same-lane lead vehicle and trigger one sampled braking event."""

    def __init__(
        self,
        config: LeadBrakeScenarioConfig,
        sampler: ScenarioParameterSampler,
        *,
        difficulty_level: int,
    ) -> None:
        self._config = config
        self._sampler = sampler
        self._difficulty_level = difficulty_level
        self._brake_triggered = False

    def reset(self, environment: DrivingEnvironment, *, seeds: EpisodeSeeds) -> ScenarioState:
        del environment
        self._brake_triggered = False
        deceleration_range = (
            self._config.mild_deceleration_mps2
            if self._difficulty_level == 1
            else self._config.severe_deceleration_mps2
        )
        return ScenarioState(
            "lead_brake",
            seeds,
            {
                "difficulty_level": self._difficulty_level,
                "initial_gap_m": self._sampler.uniform(
                    "initial_gap_m",
                    self._config.initial_gap_m.minimum,
                    self._config.initial_gap_m.maximum,
                ),
                "speed_fraction": self._sampler.uniform(
                    "speed_fraction",
                    self._config.speed_fraction.minimum,
                    self._config.speed_fraction.maximum,
                ),
                "trigger_s": self._sampler.uniform(
                    "trigger_s", self._config.trigger_s.minimum, self._config.trigger_s.maximum
                ),
                "deceleration_mps2": self._sampler.uniform(
                    "deceleration_mps2", deceleration_range.minimum, deceleration_range.maximum
                ),
                "survival_s": self._config.survival_s,
                "actor_id": _LEAD_ACTOR_ID,
            },
        )

    def after_simulator_reset(
        self, environment: DrivingEnvironment, state: ScenarioState
    ) -> ScenarioState:
        geometry = environment.scenario_road_geometry()
        initial_gap = self._parameter_float(state, "initial_gap_m")
        lead_speed = geometry.ego_speed_mps * self._parameter_float(state, "speed_fraction")
        actor_id = self._parameter_str(state, "actor_id")
        spawned_id = environment.scenario_spawn_lane_vehicle(
            LaneVehicleSpawn(
                actor_id,
                geometry.ego_lane_index,
                geometry.ego_longitudinal_m + initial_gap,
                geometry.ego_lateral_m,
                lead_speed,
            )
        )
        if spawned_id != actor_id:
            raise RuntimeError("scenario Actor spawn returned an unexpected ID")
        trigger_step = ceil(
            self._parameter_float(state, "trigger_s") / geometry.decision_interval_s
        )
        success_step = trigger_step + ceil(
            self._parameter_float(state, "survival_s") / geometry.decision_interval_s
        )
        return replace(
            state,
            parameters={
                **state.parameters,
                "lead_speed_mps": lead_speed,
                "trigger_step": trigger_step,
                "success_step": success_step,
            },
        )

    def before_step(
        self, environment: DrivingEnvironment, state: ScenarioState, *, step_index: int
    ) -> ScenarioState:
        actor_id = self._require_spawned_actor(environment, state)
        if not self._brake_triggered and step_index >= self._parameter_int(state, "trigger_step"):
            environment.scenario_command_actor(
                actor_id,
                ActorCommand.longitudinal(-self._parameter_float(state, "deceleration_mps2")),
            )
            self._brake_triggered = True
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

    def _require_spawned_actor(self, environment: DrivingEnvironment, state: ScenarioState) -> str:
        actor_id = self._parameter_str(state, "actor_id")
        if actor_id not in environment.scenario_actor_ids():
            raise RuntimeError(f"missing scenario Actor: {actor_id}")
        return actor_id
