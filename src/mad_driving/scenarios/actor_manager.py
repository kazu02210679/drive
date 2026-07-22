"""Thin MetaDrive owner for scripted Phase 5 scenario actors."""

from __future__ import annotations

from math import cos, sin
from typing import Any

from metadrive.component.vehicle.vehicle_type import (  # type: ignore[import-untyped]
    DefaultVehicle,
    StaticDefaultVehicle,
)
from metadrive.manager.base_manager import BaseManager  # type: ignore[import-untyped]

from mad_driving.scenarios.actors import (
    ActorCommand,
    KinematicActorSpawn,
    LaneVehicleSpawn,
    ScenarioActorState,
    StaticOccluderSpawn,
)


class ScenarioActorManager(BaseManager):  # type: ignore[misc]
    """Own, command, refresh, and clean simulator objects for scenario runtimes."""

    PRIORITY = 20

    def __init__(self, *, engine: Any | None = None) -> None:
        self._engine_override = engine
        self._pending_commands: dict[str, ActorCommand] = {}
        self._states: dict[str, ScenarioActorState] = {}
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
        self._set_speed(actor_id, spawn.speed_mps)
        return actor_id

    def spawn_crossing_actor(self, spawn: KinematicActorSpawn) -> str:
        """Spawn and own one kinematic actor at a world-relative pose."""

        actor_id = self._spawn(
            spawn.actor_id,
            DefaultVehicle,
            position=spawn.position_xy_m,
            heading=spawn.heading_rad,
        )
        actor = self._require_actor(actor_id)
        setter = getattr(actor, "set_velocity", None)
        if callable(setter):
            setter(spawn.velocity_xy_mps)
        return actor_id

    def spawn_occluder(self, spawn: StaticOccluderSpawn) -> str:
        """Spawn and own one static vehicle-shaped occluder."""

        actor_id = self._spawn(
            spawn.actor_id,
            StaticDefaultVehicle,
            position=spawn.position_xy_m,
            heading=spawn.heading_rad,
        )
        setter = getattr(self._require_actor(actor_id), "set_static", None)
        if callable(setter):
            setter(True)
        return actor_id

    def command_actor(self, actor_id: str, command: ActorCommand) -> None:
        """Queue a command for the next MetaDrive manager before-step hook."""

        self._require_actor(actor_id)
        if not isinstance(command, ActorCommand):
            raise TypeError("scenario actor command must be an ActorCommand")
        self._pending_commands[actor_id] = command

    def actor_state(self, actor_id: str) -> ScenarioActorState:
        """Return the last refreshed finite state for one owned actor."""

        self._require_actor(actor_id)
        self._refresh_state(actor_id)
        return self._states[actor_id]

    def actor_ids(self) -> tuple[str, ...]:
        """Return scenario actor identifiers in spawn order."""

        return tuple(self.spawned_objects)

    def before_step(self, *args: object, **kwargs: object) -> dict[str, object]:
        """Apply queued commands immediately before simulator advancement."""

        del args, kwargs
        for actor_id, command in tuple(self._pending_commands.items()):
            actor = self._require_actor(actor_id)
            setter = getattr(actor, "set_longitudinal_acceleration", None)
            if callable(setter):
                setter(command)
            else:
                speed = float(getattr(actor, "speed", 0.0))
                next_speed = max(
                    0.0,
                    speed + command.longitudinal_acceleration_mps2 * self._decision_interval_s(),
                )
                heading = float(getattr(actor, "heading_theta", 0.0))
                actor.set_velocity((cos(heading), sin(heading)), next_speed)
        self._pending_commands.clear()
        return {}

    def after_step(self, *args: object, **kwargs: object) -> dict[str, object]:
        """Refresh finite actor states after physics has advanced."""

        del args, kwargs
        for actor_id in self.actor_ids():
            self._refresh_state(actor_id)
        return {}

    def before_reset(self) -> None:
        """Discard pending data and let BaseManager clear all owned objects."""

        self._pending_commands.clear()
        self._states.clear()
        super().before_reset()

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

    def _refresh_state(self, actor_id: str) -> None:
        actor = self._require_actor(actor_id)
        position = tuple(float(value) for value in actor.position[:2])
        velocity = tuple(float(value) for value in actor.velocity[:2])
        last_velocity = getattr(actor, "last_velocity", actor.velocity)
        previous = tuple(float(value) for value in last_velocity[:2])
        decision_interval_s = self._decision_interval_s()
        self._states[actor_id] = ScenarioActorState(
            actor_id=actor_id,
            position_xy_m=(position[0], position[1]),
            velocity_xy_mps=(velocity[0], velocity[1]),
            acceleration_xy_mps2=(
                (velocity[0] - previous[0]) / decision_interval_s,
                (velocity[1] - previous[1]) / decision_interval_s,
            ),
            heading_rad=float(actor.heading_theta),
        )

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
