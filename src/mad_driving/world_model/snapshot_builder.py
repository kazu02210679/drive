"""Build immutable, SI-unit snapshots from live MetaDrive state."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, pi, sin
from typing import TYPE_CHECKING, Any

from mad_driving.interfaces import (
    ActorState,
    CollisionKind,
    EgoState,
    PrivilegedWorldState,
    RoadContext,
    SceneFrame,
    SceneObservation,
)
from mad_driving.interfaces.actor_state import ActorType
from mad_driving.scenarios.seeding import EpisodeSeeds
from mad_driving.world_model.validation import (
    ConfigReader,
    decision_interval_s,
    finite_float,
    xy_pair,
)

if TYPE_CHECKING:
    from mad_driving.scenarios.runtime import (
        ScenarioObservationContext,
        ScenarioStepResult,
    )


class SceneSnapshotBuilder:
    """Translate MetaDrive runtime objects into the project boundary model."""

    def build(
        self,
        env: Any,
        *,
        step_index: int,
        seeds: EpisodeSeeds,
        context: ScenarioObservationContext,
        scenario_result: ScenarioStepResult,
        raw_info: Mapping[str, object],
        previous_executed_action: int,
        previous_shield_intervention: bool,
    ) -> SceneFrame:
        config = self._config(env)
        interval_s = decision_interval_s(config)
        ego_vehicle = env.agent if hasattr(env, "agent") else env.vehicle
        ego_position = xy_pair("ego.position", ego_vehicle.position)
        ego_velocity = xy_pair("ego.velocity", ego_vehicle.velocity)
        ego_last_velocity = xy_pair(
            "ego.last_velocity", getattr(ego_vehicle, "last_velocity", ego_velocity)
        )
        heading = self._normalized_heading("ego.heading_theta", ego_vehicle.heading_theta)
        acceleration_xy = self._acceleration(ego_velocity, ego_last_velocity, interval_s)
        longitudinal_acceleration = acceleration_xy[0] * cos(heading) + acceleration_xy[1] * sin(
            heading
        )
        lane = self._current_lane(ego_vehicle)

        ego = EgoState(
            position_xy_m=ego_position,
            speed_mps=finite_float("ego.speed_mps", ego_vehicle.speed),
            acceleration_mps2=longitudinal_acceleration,
            heading_rad=heading,
            lane_offset_m=self._lane_offset(lane, ego_position),
            route_progress=self._route_progress(ego_vehicle),
            speed_limit_mps=self._speed_limit_mps(ego_vehicle, lane),
        )
        all_actors = self._actors(
            env,
            ego_vehicle,
            ego_position,
            heading,
            interval_s,
            config,
        )
        visible_actors = tuple(
            actor
            for actor in all_actors
            if context.visible_actor_ids is None
            or actor.actor_id in context.visible_actor_ids
        )
        observation = SceneObservation(
            step_index=step_index,
            sim_time_s=step_index * interval_s,
            scenario_id=context.scenario_id,
            seeds=seeds,
            ego=ego,
            visible_actors=visible_actors,
            occlusion_regions=context.occlusion_regions,
            road_context=RoadContext(
                stop_required=context.stop_required,
                distance_to_conflict_point_m=context.distance_to_conflict_point_m,
                intersection_entry_prohibited=context.intersection_entry_prohibited,
            ),
            previous_executed_action=previous_executed_action,
            previous_shield_intervention=previous_shield_intervention,
        )
        privileged = PrivilegedWorldState(
            all_actors=all_actors,
            collision_occurred=self._collision_occurred(raw_info, ego_vehicle),
            collision_kind=self._collision_kind(raw_info, ego_vehicle),
            off_road=bool(raw_info.get("out_of_road", False))
            or self._off_road(ego_vehicle),
            arrived=bool(raw_info.get("arrive_dest", False)),
            scenario_success=scenario_result.success,
            scenario_failure=scenario_result.failure,
        )
        return SceneFrame(observation=observation, privileged=privileged)

    @staticmethod
    def _collision_occurred(raw_info: Mapping[str, object], vehicle: Any) -> bool:
        return any(
            bool(raw_info.get(attribute, False)) or bool(getattr(vehicle, attribute, False))
            for attribute in (
                "crash_vehicle",
                "crash_human",
                "crash_object",
                "crash_sidewalk",
                "crash_building",
            )
        )

    @staticmethod
    def _collision_kind(
        raw_info: Mapping[str, object], vehicle: Any
    ) -> CollisionKind | None:
        kinds: tuple[tuple[str, CollisionKind], ...] = (
            ("crash_vehicle", "vehicle"),
            ("crash_human", "crossing_actor"),
            ("crash_object", "object"),
            ("crash_sidewalk", "sidewalk"),
            ("crash_building", "building"),
        )
        for attribute, kind in kinds:
            if bool(raw_info.get(attribute, False)) or bool(
                getattr(vehicle, attribute, False)
            ):
                return kind
        return None

    @staticmethod
    def _off_road(vehicle: Any) -> bool:
        on_lane = getattr(vehicle, "on_lane", None)
        return on_lane is not None and not bool(on_lane)

    @staticmethod
    def _config(env: Any) -> ConfigReader:
        config = getattr(env, "config", {})
        if not callable(getattr(config, "get", None)):
            raise TypeError("env.config must provide get(key, default)")
        return config

    def _actors(
        self,
        env: Any,
        ego_vehicle: Any,
        ego_position: tuple[float, float],
        ego_heading: float,
        interval_s: float,
        config: ConfigReader,
    ) -> tuple[ActorState, ...]:
        ego_lane = self._current_lane(ego_vehicle)
        ego_lane_index = self._canonical_lane_index(ego_vehicle, ego_lane)
        actors: list[ActorState] = []
        for object_key, simulator_object in env.engine.get_objects().items():
            if simulator_object is ego_vehicle or not self._has_actor_state(simulator_object):
                continue
            position = xy_pair(f"actor[{object_key}].position", simulator_object.position)
            velocity = xy_pair(f"actor[{object_key}].velocity", simulator_object.velocity)
            last_velocity = xy_pair(
                f"actor[{object_key}].last_velocity",
                getattr(simulator_object, "last_velocity", velocity),
            )
            dx = position[0] - ego_position[0]
            dy = position[1] - ego_position[1]
            actor_id = str(getattr(simulator_object, "name", object_key))
            actor_lane = self._current_lane(simulator_object)
            actors.append(
                ActorState(
                    actor_id=actor_id,
                    actor_type=self._actor_type(simulator_object),
                    position_xy_m=position,
                    velocity_xy_mps=velocity,
                    acceleration_xy_mps2=self._acceleration(velocity, last_velocity, interval_s),
                    heading_rad=self._normalized_heading(
                        f"actor[{actor_id}].heading_theta", simulator_object.heading_theta
                    ),
                    length_m=finite_float(f"actor[{actor_id}].length", simulator_object.LENGTH),
                    width_m=finite_float(f"actor[{actor_id}].width", simulator_object.WIDTH),
                    relative_longitudinal_m=cos(ego_heading) * dx + sin(ego_heading) * dy,
                    relative_lateral_m=-sin(ego_heading) * dx + cos(ego_heading) * dy,
                    same_lane=self._same_lane(
                        ego_lane_index,
                        simulator_object,
                        actor_lane,
                        position,
                        config,
                    ),
                    visible=True,
                    occluded=False,
                )
            )
        return tuple(sorted(actors, key=lambda actor: actor.actor_id))

    @staticmethod
    def _normalized_heading(name: str, value: Any) -> float:
        heading = finite_float(name, value)
        return (heading + pi) % (2.0 * pi) - pi

    @staticmethod
    def _canonical_lane_index(vehicle: Any, lane: Any | None) -> tuple[object, ...] | None:
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is None and lane is not None:
            lane_index = getattr(lane, "index", None)
        if isinstance(lane_index, tuple | list):
            return tuple(lane_index)
        return None

    def _same_lane(
        self,
        ego_lane_index: tuple[object, ...] | None,
        actor_vehicle: Any,
        actor_lane: Any | None,
        actor_position: tuple[float, float],
        config: ConfigReader,
    ) -> bool:
        if ego_lane_index is None:
            return False
        if self._canonical_lane_index(actor_vehicle, actor_lane) != ego_lane_index:
            return False
        if actor_lane is None:
            return False
        _, lateral_position = actor_lane.local_coordinates(actor_position)
        lateral_position = finite_float("actor.lane_lateral_m", lateral_position)
        return abs(lateral_position) <= self._lane_width_m(actor_lane, config) / 2.0

    @staticmethod
    def _lane_width_m(lane: Any, config: ConfigReader) -> float:
        for attribute in ("width", "WIDTH"):
            exposed_width = getattr(lane, attribute, None)
            if exposed_width is not None:
                width = finite_float(f"lane.{attribute}", exposed_width)
                if width <= 0.0:
                    raise ValueError(f"lane.{attribute} must be positive")
                return width
        map_config = config.get("map_config")
        if map_config is None or not callable(getattr(map_config, "get", None)):
            raise ValueError("map_config.lane_width must be configured")
        lane_width = map_config.get("lane_width")
        if lane_width is None:
            raise ValueError("map_config.lane_width must be configured")
        width = finite_float("map_config.lane_width", lane_width)
        if width <= 0.0:
            raise ValueError("map_config.lane_width must be positive")
        return width

    @staticmethod
    def _has_actor_state(simulator_object: Any) -> bool:
        return all(
            hasattr(simulator_object, attribute)
            for attribute in ("position", "velocity", "heading_theta", "LENGTH", "WIDTH")
        )

    @staticmethod
    def _actor_type(simulator_object: Any) -> ActorType:
        class_name = type(simulator_object).__name__.lower()
        if "pedestrian" in class_name or "cyclist" in class_name:
            return "crossing_actor"
        if hasattr(simulator_object, "navigation") or "vehicle" in class_name:
            return "vehicle"
        return "obstacle"

    @staticmethod
    def _acceleration(
        velocity: tuple[float, float],
        last_velocity: tuple[float, float],
        interval_s: float,
    ) -> tuple[float, float]:
        return (
            (velocity[0] - last_velocity[0]) / interval_s,
            (velocity[1] - last_velocity[1]) / interval_s,
        )

    @staticmethod
    def _current_lane(vehicle: Any) -> Any | None:
        navigation = getattr(vehicle, "navigation", None)
        if navigation is not None:
            return getattr(navigation, "current_lane", None)
        return getattr(vehicle, "lane", None)

    @staticmethod
    def _lane_offset(lane: Any | None, position: tuple[float, float]) -> float:
        if lane is None:
            return 0.0
        _, lateral = lane.local_coordinates(position)
        return finite_float("ego.lane_offset_m", lateral)

    @staticmethod
    def _route_progress(vehicle: Any) -> float:
        navigation = getattr(vehicle, "navigation", None)
        if navigation is None:
            return 0.0
        value = finite_float("ego.route_progress", getattr(navigation, "route_completion", 0.0))
        return min(max(value, 0.0), 1.0)

    @staticmethod
    def _speed_limit_mps(vehicle: Any, lane: Any | None) -> float:
        if lane is not None and hasattr(lane, "speed_limit"):
            return finite_float("ego.speed_limit_mps", lane.speed_limit) / 3.6
        return finite_float("ego.speed_limit_mps", getattr(vehicle, "max_speed_m_s", 0.0))
