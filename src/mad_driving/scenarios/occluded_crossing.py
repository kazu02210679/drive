"""Seeded cyclist crossing hidden behind a static logical occlusion boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import atan2, ceil, hypot
from typing import TYPE_CHECKING

from mad_driving.config.models import OccludedCrossingScenarioConfig
from mad_driving.interfaces import OcclusionRegion
from mad_driving.scenarios.actors import (
    KinematicActorSpawn,
    LaneVehicleSpawn,
    StaticOccluderSpawn,
    VelocityCommand,
)
from mad_driving.scenarios.parameters import ScenarioParameterSampler
from mad_driving.scenarios.runtime import (
    ScenarioObservationContext,
    ScenarioState,
    ScenarioStepResult,
    ScenarioTransition,
    typed_collision_flags,
)
from mad_driving.scenarios.seeding import EpisodeSeeds

if TYPE_CHECKING:
    from mad_driving.envs.multi_agent_speed_env import DrivingEnvironment


_CYCLIST_ID = "crossing-cyclist"
_OCCLUDER_ID = "static-occluder"
_LEAD_ID = "crossing-lead"
_OCCLUDER_LENGTH_M = 5.0
_OCCLUDER_WIDTH_M = 2.0
_CYCLIST_HALF_WIDTH_M = 0.2


class OccludedCrossingRuntime:
    """Create a Level-3 crossing cyclist and reveal it at a fixed lateral boundary."""

    def __init__(
        self,
        config: OccludedCrossingScenarioConfig,
        sampler: ScenarioParameterSampler,
        *,
        difficulty_level: int,
    ) -> None:
        self._config = config
        self._sampler = sampler
        self._difficulty_level = difficulty_level
        self._environment: DrivingEnvironment | None = None

    def reset(self, environment: DrivingEnvironment, *, seeds: EpisodeSeeds) -> ScenarioState:
        del environment
        self._environment = None
        return ScenarioState(
            "occluded_crossing",
            seeds,
            {
                "difficulty_level": self._difficulty_level,
                "conflict_distance_m": self._sampler.uniform(
                    "conflict_distance_m",
                    self._config.conflict_distance_m.minimum,
                    self._config.conflict_distance_m.maximum,
                ),
                "crossing_start_offset_m": self._sampler.uniform(
                    "crossing_start_offset_m",
                    self._config.crossing_start_offset_m.minimum,
                    self._config.crossing_start_offset_m.maximum,
                ),
                "crossing_speed_mps": self._sampler.uniform(
                    "crossing_speed_mps",
                    self._config.crossing_speed_mps.minimum,
                    self._config.crossing_speed_mps.maximum,
                ),
                "trigger_s": self._sampler.uniform(
                    "trigger_s", self._config.trigger_s.minimum, self._config.trigger_s.maximum
                ),
                "survival_s": self._config.survival_s,
                "reveal_lateral_m": self._config.reveal_lateral_m,
                "secondary_lead_gap_m": self._sampler.uniform(
                    "secondary_lead_gap_m",
                    self._config.secondary_lead_gap_m.minimum,
                    self._config.secondary_lead_gap_m.maximum,
                ),
                "secondary_lead_speed_fraction": self._sampler.uniform(
                    "secondary_lead_speed_fraction",
                    self._config.secondary_lead_speed_fraction.minimum,
                    self._config.secondary_lead_speed_fraction.maximum,
                ),
                "cyclist_actor_id": _CYCLIST_ID,
                "occluder_actor_id": _OCCLUDER_ID,
                "lead_actor_id": _LEAD_ID,
                "cyclist_revealed": False,
                "cyclist_collision": False,
            },
        )

    def after_simulator_reset(
        self, environment: DrivingEnvironment, state: ScenarioState
    ) -> ScenarioState:
        geometry = environment.scenario_road_geometry()
        conflict_longitudinal_m = (
            geometry.ego_longitudinal_m + self._parameter_float(state, "conflict_distance_m")
        )
        conflict_point = environment.scenario_lane_position(
            geometry.ego_lane_index, conflict_longitudinal_m, 0.0
        )
        next_point = environment.scenario_lane_position(
            geometry.ego_lane_index, conflict_longitudinal_m + 1.0, 0.0
        )
        tangent = self._unit_vector(
            (next_point[0] - conflict_point[0], next_point[1] - conflict_point[1])
        )
        crossing_direction = (-tangent[1], tangent[0])
        crossing_start_offset_m = self._parameter_float(state, "crossing_start_offset_m")
        cyclist_position = self._translated(
            conflict_point, crossing_direction, crossing_start_offset_m
        )
        cyclist_velocity_direction = (-crossing_direction[0], -crossing_direction[1])
        cyclist_id = self._parameter_str(state, "cyclist_actor_id")
        spawned_id = environment.scenario_spawn_crossing_actor(
            KinematicActorSpawn(
                cyclist_id,
                "crossing_actor",
                cyclist_position,
                atan2(cyclist_velocity_direction[1], cyclist_velocity_direction[0]),
                (0.0, 0.0),
            )
        )
        if spawned_id != cyclist_id:
            raise RuntimeError("scenario Actor spawn returned an unexpected ID")

        occluder_lateral_m = geometry.lane_width_m / 2.0 + self._config.occluder_lane_edge_offset_m
        occluder_position = self._translated(
            self._translated(conflict_point, tangent, _OCCLUDER_LENGTH_M),
            crossing_direction,
            occluder_lateral_m + _OCCLUDER_WIDTH_M / 2.0,
        )
        occluder_id = self._parameter_str(state, "occluder_actor_id")
        spawned_id = environment.scenario_spawn_occluder(
            StaticOccluderSpawn(
                occluder_id,
                occluder_position,
                atan2(tangent[1], tangent[0]),
                _OCCLUDER_LENGTH_M,
                _OCCLUDER_WIDTH_M,
            )
        )
        if spawned_id != occluder_id:
            raise RuntimeError("scenario Actor spawn returned an unexpected ID")

        lead_id = self._parameter_str(state, "lead_actor_id")
        lead_speed_mps = geometry.ego_speed_mps * self._parameter_float(
            state, "secondary_lead_speed_fraction"
        )
        spawned_id = environment.scenario_spawn_lane_vehicle(
            LaneVehicleSpawn(
                lead_id,
                geometry.ego_lane_index,
                geometry.ego_longitudinal_m
                + self._parameter_float(state, "secondary_lead_gap_m"),
                geometry.ego_lateral_m,
                lead_speed_mps,
            )
        )
        if spawned_id != lead_id:
            raise RuntimeError("scenario Actor spawn returned an unexpected ID")

        self._environment = environment
        trigger_step = ceil(
            self._parameter_float(state, "trigger_s") / geometry.decision_interval_s
        )
        return replace(
            state,
            parameters={
                **state.parameters,
                "conflict_longitudinal_m": conflict_longitudinal_m,
                "conflict_point_xy_m": conflict_point,
                "crossing_direction_xy": crossing_direction,
                "cyclist_velocity_direction_xy": cyclist_velocity_direction,
                "trigger_step": trigger_step,
                "clear_lateral_m": geometry.lane_width_m / 2.0 + _CYCLIST_HALF_WIDTH_M,
                "decision_interval_s": geometry.decision_interval_s,
                "lead_speed_mps": lead_speed_mps,
            },
        )

    def before_step(
        self, environment: DrivingEnvironment, state: ScenarioState, *, step_index: int
    ) -> ScenarioState:
        cyclist_id = self._require_spawned_actor(environment, state, "cyclist_actor_id")
        if step_index == self._parameter_int(state, "trigger_step"):
            environment.scenario_command_actor(
                cyclist_id,
                VelocityCommand(
                    self._parameter_xy(state, "cyclist_velocity_direction_xy"),
                    self._parameter_float(state, "crossing_speed_mps"),
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
        cyclist_id = self._require_spawned_actor(environment, state, "cyclist_actor_id")
        self._require_spawned_actor(environment, state, "occluder_actor_id")
        self._require_spawned_actor(environment, state, "lead_actor_id")
        next_state = state
        lateral_m = self._cyclist_lateral_m(environment, state)
        is_revealed = self._parameter_bool(state, "cyclist_revealed")
        reveal_lateral_m = self._parameter_float(state, "reveal_lateral_m")
        if not is_revealed and abs(lateral_m) <= reveal_lateral_m:
            next_state = replace(
                next_state,
                parameters={**next_state.parameters, "cyclist_revealed": True},
            )
        if "cleared_step" not in state.parameters and step_index >= self._parameter_int(
            state, "trigger_step"
        ):
            if lateral_m <= -self._parameter_float(state, "clear_lateral_m"):
                next_state = replace(
                    next_state,
                    parameters={**next_state.parameters, "cleared_step": step_index},
                )
        collisions = typed_collision_flags(raw_info)
        failure = (
            self._parameter_bool(state, "cyclist_collision")
            or "crash_human" in collisions
            or environment.scenario_ego_collided_with(cyclist_id)
        )
        if failure and not self._parameter_bool(next_state, "cyclist_collision"):
            next_state = replace(
                next_state,
                parameters={**next_state.parameters, "cyclist_collision": True},
            )
        off_road = bool(raw_info.get("out_of_road", False)) or bool(raw_info.get("off_road", False))
        success = (
            not collisions
            and not off_road
            and not failure
            and self._survival_complete(next_state, step_index)
        )
        return ScenarioTransition(next_state, ScenarioStepResult(success=success, failure=failure))

    def observation_context(self, state: ScenarioState) -> ScenarioObservationContext:
        environment = self._require_environment()
        cyclist_id = self._parameter_str(state, "cyclist_actor_id")
        visible_ids = set(environment.scenario_visible_actor_ids())
        if self._parameter_bool(state, "cyclist_revealed") or abs(
            self._cyclist_lateral_m(environment, state)
        ) <= self._parameter_float(state, "reveal_lateral_m"):
            visible_ids.add(cyclist_id)
        else:
            visible_ids.discard(cyclist_id)
        geometry = environment.scenario_road_geometry()
        conflict_distance_m = (
            self._parameter_float(state, "conflict_longitudinal_m") - geometry.ego_longitudinal_m
        )
        conflict_point = self._parameter_xy(state, "conflict_point_xy_m")
        tangent = self._perpendicular(self._parameter_xy(state, "crossing_direction_xy"))
        reveal_point = self._translated(
            conflict_point,
            self._parameter_xy(state, "crossing_direction_xy"),
            self._parameter_float(state, "reveal_lateral_m"),
        )
        region = OcclusionRegion(
            "crossing-occluder",
            (
                self._translated(reveal_point, tangent, -_OCCLUDER_LENGTH_M / 2.0),
                self._translated(reveal_point, tangent, _OCCLUDER_LENGTH_M / 2.0),
            ),
        )
        return ScenarioObservationContext(
            scenario_id=state.scenario_id,
            occlusion_regions=(region,),
            distance_to_conflict_point_m=conflict_distance_m,
            visible_actor_ids=frozenset(visible_ids),
        )

    def _survival_complete(self, state: ScenarioState, step_index: int) -> bool:
        cleared_step = state.parameters.get("cleared_step")
        if isinstance(cleared_step, bool) or not isinstance(cleared_step, int):
            return False
        return step_index >= cleared_step + ceil(
            self._parameter_float(state, "survival_s")
            / self._parameter_float(state, "decision_interval_s")
        )

    def _cyclist_lateral_m(self, environment: DrivingEnvironment, state: ScenarioState) -> float:
        cyclist = environment.scenario_actor_state(self._parameter_str(state, "cyclist_actor_id"))
        conflict_point = self._parameter_xy(state, "conflict_point_xy_m")
        direction = self._parameter_xy(state, "crossing_direction_xy")
        return (
            (cyclist.position_xy_m[0] - conflict_point[0]) * direction[0]
            + (cyclist.position_xy_m[1] - conflict_point[1]) * direction[1]
        )

    def _require_spawned_actor(
        self, environment: DrivingEnvironment, state: ScenarioState, parameter: str
    ) -> str:
        actor_id = self._parameter_str(state, parameter)
        if actor_id not in environment.scenario_actor_ids():
            raise RuntimeError(f"missing scenario Actor: {actor_id}")
        return actor_id

    def _require_environment(self) -> DrivingEnvironment:
        if self._environment is None:
            raise RuntimeError("OccludedCrossingRuntime.after_simulator_reset must be called first")
        return self._environment

    @staticmethod
    def _unit_vector(vector: tuple[float, float]) -> tuple[float, float]:
        magnitude = hypot(*vector)
        if magnitude == 0.0:
            raise RuntimeError("ego lane tangent must be non-zero")
        return vector[0] / magnitude, vector[1] / magnitude

    @staticmethod
    def _perpendicular(vector: tuple[float, float]) -> tuple[float, float]:
        return -vector[1], vector[0]

    @staticmethod
    def _translated(
        origin: tuple[float, float], direction: tuple[float, float], distance_m: float
    ) -> tuple[float, float]:
        return origin[0] + direction[0] * distance_m, origin[1] + direction[1] * distance_m

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
    def _parameter_bool(state: ScenarioState, name: str) -> bool:
        value = state.parameters[name]
        if not isinstance(value, bool):
            raise RuntimeError(f"scenario parameter {name} must be a boolean")
        return value

    @staticmethod
    def _parameter_xy(state: ScenarioState, name: str) -> tuple[float, float]:
        value = state.parameters[name]
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or isinstance(value[0], bool)
            or isinstance(value[1], bool)
            or not isinstance(value[0], float)
            or not isinstance(value[1], float)
        ):
            raise RuntimeError(f"scenario parameter {name} must be an XY pair")
        return value
