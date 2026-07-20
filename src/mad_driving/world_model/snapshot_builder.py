"""Build immutable, SI-unit snapshots from live MetaDrive state."""

from __future__ import annotations

from math import cos, sin
from typing import Any

from mad_driving.interfaces import ActorState, EgoState, SceneSnapshot
from mad_driving.interfaces.actor_state import ActorType
from mad_driving.world_model.validation import (
    ConfigReader,
    decision_interval_s,
    finite_float,
    xy_pair,
)


class SceneSnapshotBuilder:
    """Translate MetaDrive runtime objects into the project boundary model."""

    def build(
        self,
        env: Any,
        *,
        step_index: int,
        scenario_id: str,
        seed: int,
        previous_action: int,
        previous_shield_intervention: bool,
        stop_required: bool = False,
        occlusion_present: bool = False,
        distance_to_conflict_point_m: float | None = None,
        intersection_entry_prohibited: bool = False,
    ) -> SceneSnapshot:
        config = self._config(env)
        interval_s = decision_interval_s(config)
        ego_vehicle = env.agent if hasattr(env, "agent") else env.vehicle
        ego_position = xy_pair("ego.position", ego_vehicle.position)
        ego_velocity = xy_pair("ego.velocity", ego_vehicle.velocity)
        ego_last_velocity = xy_pair(
            "ego.last_velocity", getattr(ego_vehicle, "last_velocity", ego_velocity)
        )
        heading = finite_float("ego.heading_theta", ego_vehicle.heading_theta)
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
        actors = self._actors(env, ego_vehicle, ego_position, heading, interval_s)

        return SceneSnapshot(
            step_index=step_index,
            sim_time_s=step_index * interval_s,
            scenario_id=scenario_id,
            seed=seed,
            ego=ego,
            actors=actors,
            stop_required=stop_required,
            occlusion_present=occlusion_present,
            distance_to_conflict_point_m=distance_to_conflict_point_m,
            previous_action=previous_action,
            previous_shield_intervention=previous_shield_intervention,
            collision_occurred=False,
            off_road=False,
            intersection_entry_prohibited=intersection_entry_prohibited,
        )

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
    ) -> tuple[ActorState, ...]:
        ego_lane_index = getattr(ego_vehicle, "lane_index", None)
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
            actors.append(
                ActorState(
                    actor_id=actor_id,
                    actor_type=self._actor_type(simulator_object),
                    position_xy_m=position,
                    velocity_xy_mps=velocity,
                    acceleration_xy_mps2=self._acceleration(velocity, last_velocity, interval_s),
                    heading_rad=finite_float(
                        f"actor[{actor_id}].heading_theta", simulator_object.heading_theta
                    ),
                    length_m=finite_float(f"actor[{actor_id}].length", simulator_object.LENGTH),
                    width_m=finite_float(f"actor[{actor_id}].width", simulator_object.WIDTH),
                    relative_longitudinal_m=cos(ego_heading) * dx + sin(ego_heading) * dy,
                    relative_lateral_m=-sin(ego_heading) * dx + cos(ego_heading) * dy,
                    same_lane=(
                        ego_lane_index is not None
                        and getattr(simulator_object, "lane_index", None) == ego_lane_index
                    ),
                    visible=True,
                    occluded=False,
                )
            )
        return tuple(sorted(actors, key=lambda actor: actor.actor_id))

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
