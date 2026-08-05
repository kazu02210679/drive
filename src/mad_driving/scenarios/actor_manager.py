"""Thin MetaDrive owner for scripted Phase 5 scenario actors."""

from __future__ import annotations

from math import cos, sin
from typing import Any

from metadrive.component.static_object.traffic_object import (  # type: ignore[import-untyped]
    TrafficBarrier,
)
from metadrive.component.traffic_participants.cyclist import Cyclist  # type: ignore[import-untyped]
from metadrive.component.vehicle.vehicle_type import (  # type: ignore[import-untyped]
    DefaultVehicle,
)
from metadrive.manager.base_manager import BaseManager  # type: ignore[import-untyped]
from metadrive.utils.utils import get_object_from_node  # type: ignore[import-untyped]

from mad_driving.scenarios.actors import (
    ActorCommand,
    KinematicActorSpawn,
    LanePoseCommand,
    LaneVehicleSpawn,
    ScenarioActorCommand,
    ScenarioActorState,
    StaticOccluderSpawn,
    VelocityCommand,
)


class ScenarioOccluder(TrafficBarrier):  # type: ignore[misc]
    """Supported MetaDrive traffic object with scenario-configured dimensions."""

    def __init__(
        self,
        position: tuple[float, float],
        heading_theta: float,
        *,
        length_m: float,
        width_m: float,
        lane: Any | None = None,
        static: bool = True,
        random_seed: int | None = None,
        name: str | None = None,
    ) -> None:
        self._scenario_length_m = length_m
        self._scenario_width_m = width_m
        super().__init__(
            position,
            heading_theta,
            lane=lane,
            static=static,
            random_seed=random_seed,
            name=name,
        )

    @property
    def LENGTH(self) -> float:
        return self._scenario_length_m

    @property
    def WIDTH(self) -> float:
        return self._scenario_width_m


class ScenarioActorManager(BaseManager):  # type: ignore[misc]
    """Own, command, refresh, and clean simulator objects for scenario runtimes."""

    PRIORITY = 20

    def __init__(self, *, engine: Any | None = None) -> None:
        self._engine_override = engine
        self._pending_commands: dict[str, ScenarioActorCommand] = {}
        self._states: dict[str, ScenarioActorState] = {}
        self._pre_step_positions: dict[str, tuple[float, float]] = {}
        self._pre_step_velocities: dict[str, tuple[float, float]] = {}
        self._pre_step_state_velocities: dict[str, tuple[float, float]] = {}
        self._lane_pose_actor_ids: set[str] = set()
        self._lane_reference_indices: dict[str, tuple[str, str, int]] = {}
        if engine is None:
            super().__init__()
        else:
            self.spawned_objects: dict[str, Any] = {}

    @property
    def engine(self) -> Any:
        if self._engine_override is not None:
            return self._engine_override
        return super().engine

    def spawn_lane_vehicle(self, spawn: LaneVehicleSpawn) -> str:
        """Spawn and own one vehicle using exact lane-relative placement."""

        actor_id = self._spawn(
            spawn.actor_id,
            DefaultVehicle,
            vehicle_config={
                "spawn_lane_index": spawn.lane_index,
                "spawn_longitude": spawn.longitudinal_m,
                "spawn_lateral": spawn.lateral_m,
            },
        )
        self._lane_reference_indices[actor_id] = spawn.lane_index
        self._set_speed(actor_id, spawn.speed_mps)
        self._refresh_state(actor_id)
        return actor_id

    def spawn_crossing_actor(self, spawn: KinematicActorSpawn) -> str:
        """Spawn and own one kinematic actor at a world-relative pose."""

        if spawn.actor_kind != "crossing_actor":
            raise ValueError("crossing actor spawn must have actor_kind='crossing_actor'")
        actor_id = self._spawn(
            spawn.actor_id,
            Cyclist,
            position=spawn.position_xy_m,
            heading_theta=spawn.heading_rad,
        )
        actor = self._require_actor(actor_id)
        setter = getattr(actor, "set_velocity", None)
        if callable(setter):
            setter(spawn.velocity_xy_mps)
        self._refresh_state(actor_id)
        return actor_id

    def spawn_occluder(self, spawn: StaticOccluderSpawn) -> str:
        """Spawn and own one static traffic-object occluder."""

        actor_id = self._spawn(
            spawn.actor_id,
            ScenarioOccluder,
            position=spawn.position_xy_m,
            heading_theta=spawn.heading_rad,
            length_m=spawn.length_m,
            width_m=spawn.width_m,
            static=True,
        )
        return actor_id

    def command_actor(self, actor_id: str, command: ScenarioActorCommand) -> None:
        """Queue a command for the next MetaDrive manager before-step hook."""

        self._require_actor(actor_id)
        if not isinstance(command, ActorCommand | LanePoseCommand | VelocityCommand):
            raise TypeError("scenario actor command must be a supported command type")
        self._pending_commands[actor_id] = command

    def actor_state(self, actor_id: str) -> ScenarioActorState:
        """Return the last refreshed finite state for one owned actor."""

        self._require_actor(actor_id)
        return self._states[actor_id]

    def actor_ids(self) -> tuple[str, ...]:
        """Return scenario actor identifiers in spawn order."""

        return tuple(self.spawned_objects)

    def ego_collided_with(self, ego_vehicle: Any, actor_id: str) -> bool:
        """Return whether the ego chassis currently contacts one owned scenario actor."""

        actor = self._require_actor(actor_id)
        ego_node = ego_vehicle.chassis.node()
        contacts = self.engine.physics_world.dynamic_world.contactTest(ego_node, True)
        for contact in contacts.getContacts():
            first = contact.getNode0()
            second = contact.getNode1()
            if first == ego_node:
                other = second
            elif second == ego_node:
                other = first
            else:
                continue
            if get_object_from_node(other) is actor:
                return True
        return False

    def before_step(self, *args: object, **kwargs: object) -> dict[str, object]:
        """Apply queued commands immediately before simulator advancement."""

        del args, kwargs
        self._pre_step_positions = {
            actor_id: self._position_xy(self._require_actor(actor_id))
            for actor_id in self.actor_ids()
        }
        self._pre_step_velocities = {
            actor_id: self._velocity_xy(self._require_actor(actor_id))
            for actor_id in self.actor_ids()
        }
        self._pre_step_state_velocities = {
            actor_id: self._states[actor_id].velocity_xy_mps for actor_id in self.actor_ids()
        }
        self._lane_pose_actor_ids.clear()
        for actor_id, command in tuple(self._pending_commands.items()):
            actor = self._require_actor(actor_id)
            if isinstance(command, ActorCommand):
                setter = getattr(actor, "set_longitudinal_acceleration", None)
                if callable(setter):
                    setter(command)
                else:
                    speed = float(getattr(actor, "speed", 0.0))
                    next_speed = max(
                        0.0,
                        speed
                        + command.longitudinal_acceleration_mps2 * self._decision_interval_s(),
                    )
                    heading = float(getattr(actor, "heading_theta", 0.0))
                    actor.set_velocity((cos(heading), sin(heading)), next_speed)
            elif isinstance(command, LanePoseCommand):
                lane = self.engine.current_map.road_network.get_lane(command.lane_index)
                actor.set_position(lane.position(command.longitudinal_m, command.lateral_m))
                self._lane_reference_indices[actor_id] = command.lane_index
                self._lane_pose_actor_ids.add(actor_id)
            else:
                actor.set_velocity(command.direction_xy, command.speed_mps)
        self._pending_commands.clear()
        return {}

    def after_step(self, *args: object, **kwargs: object) -> dict[str, object]:
        """Refresh finite actor states after physics has advanced."""

        del args, kwargs
        for actor_id in self.actor_ids():
            lane_pose_applied = actor_id in self._lane_pose_actor_ids
            self._refresh_state(
                actor_id,
                previous_position=(
                    self._pre_step_positions.get(actor_id) if lane_pose_applied else None
                ),
                previous_velocity=(
                    self._pre_step_state_velocities.get(actor_id)
                    if lane_pose_applied
                    else self._pre_step_velocities.get(actor_id)
                ),
            )
        self._pre_step_positions.clear()
        self._pre_step_velocities.clear()
        self._pre_step_state_velocities.clear()
        self._lane_pose_actor_ids.clear()
        return {}

    def before_reset(self) -> None:
        """Destroy owned scenario actors so no episode state can be recycled."""

        self._pending_commands.clear()
        self._states.clear()
        self._pre_step_positions.clear()
        self._pre_step_velocities.clear()
        self._pre_step_state_velocities.clear()
        self._lane_pose_actor_ids.clear()
        self._lane_reference_indices.clear()
        self.clear_objects(list(self.spawned_objects), force_destroy=True)
        self.spawned_objects = {}

    def _spawn(self, actor_id: str, object_class: type[Any], **kwargs: object) -> str:
        if actor_id in self.spawned_objects:
            raise ValueError(f"duplicate scenario Actor ID: {actor_id}")
        actor = self.spawn_object(object_class, name=actor_id, **kwargs)
        if actor.id != actor_id:
            raise RuntimeError("scenario Actor spawn returned an unexpected ID")
        self._refresh_state(actor_id)
        return actor_id

    def _require_actor(self, actor_id: str) -> Any:
        actor = self.spawned_objects.get(actor_id)
        if actor is None:
            raise KeyError(f"unknown scenario Actor: {actor_id}")
        if actor_id not in self.engine.get_objects([actor_id]):
            raise RuntimeError(f"missing scenario Actor: {actor_id}")
        return actor

    def _refresh_state(
        self,
        actor_id: str,
        *,
        previous_position: tuple[float, float] | None = None,
        previous_velocity: tuple[float, float] | None = None,
    ) -> None:
        actor = self._require_actor(actor_id)
        position = self._position_xy(actor)
        decision_interval_s = self._decision_interval_s()
        if previous_position is None:
            velocity = self._velocity_xy(actor)
        else:
            velocity = (
                (position[0] - previous_position[0]) / decision_interval_s,
                (position[1] - previous_position[1]) / decision_interval_s,
            )
        if previous_velocity is None:
            previous = velocity
        else:
            previous = previous_velocity
        self._states[actor_id] = ScenarioActorState(
            actor_id=actor_id,
            position_xy_m=(position[0], position[1]),
            velocity_xy_mps=(velocity[0], velocity[1]),
            acceleration_xy_mps2=(
                (velocity[0] - previous[0]) / decision_interval_s,
                (velocity[1] - previous[1]) / decision_interval_s,
            ),
            heading_rad=float(actor.heading_theta),
            lane_index=self._lane_reference_indices.get(actor_id),
        )

    @staticmethod
    def _position_xy(actor: Any) -> tuple[float, float]:
        position = tuple(float(value) for value in actor.position[:2])
        return position[0], position[1]

    @staticmethod
    def _velocity_xy(actor: Any) -> tuple[float, float]:
        velocity = tuple(float(value) for value in actor.velocity[:2])
        return velocity[0], velocity[1]

    def _set_speed(self, actor_id: str, speed_mps: float) -> None:
        actor = self._require_actor(actor_id)
        setter = getattr(actor, "set_velocity", None)
        if callable(setter):
            heading = float(getattr(actor, "heading_theta", 0.0))
            setter((cos(heading), sin(heading)), speed_mps)

    def _decision_interval_s(self) -> float:
        config = self.engine.global_config
        interval = float(config["physics_world_step_size"]) * int(config["decision_repeat"])
        if interval <= 0.0:
            raise RuntimeError("scenario Actor decision interval must be positive")
        return interval
