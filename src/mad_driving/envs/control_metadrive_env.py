"""MetaDrive environment binding for the four-action control Policy."""

from collections.abc import Mapping
from numbers import Integral
from typing import Any, cast

from mad_driving.config.models import ControlConfig
from mad_driving.control import LaneKeepingLongitudinalPolicy
from mad_driving.envs.multi_agent_speed_env import DrivingEnvironment
from mad_driving.scenarios import (
    ActorCommand,
    KinematicActorSpawn,
    LaneVehicleSpawn,
    RoadGeometry,
    ScenarioActorManager,
    ScenarioActorState,
    StaticOccluderSpawn,
)


def create_control_metadrive_env(
    config: dict[str, object],
    control_config: ControlConfig,
) -> DrivingEnvironment:
    """Construct MetaDrive with the custom discrete lane-keeping Policy."""

    from metadrive import MetaDriveEnv  # type: ignore[import-untyped]

    class ControlMetaDriveEnv(MetaDriveEnv):  # type: ignore[misc]
        @classmethod
        def default_config(cls) -> Any:
            defaults = super().default_config()
            defaults.update(
                {
                    "agent_policy": LaneKeepingLongitudinalPolicy,
                    "control_config": control_config.model_dump(),
                },
                allow_add_new_key=True,
            )
            return defaults

        def reset(self, seed: int | None = None) -> Any:
            if seed is not None:
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise TypeError("scenario index must be an integer")
                range_stop = self.start_index + self.num_scenarios
                if seed < self.start_index or seed >= range_stop:
                    raise ValueError(
                        "scenario index is outside configured scenario range "
                        f"[{self.start_index}, {range_stop}): {seed}"
                    )

            observation, raw_info = super().reset(seed=seed)
            if not isinstance(raw_info, Mapping):
                raise TypeError("MetaDrive reset info must be a mapping")
            info = dict(raw_info)
            reported = info.get("env_seed")
            current = getattr(self, "current_seed", None)
            candidates = tuple(value for value in (reported, current) if value is not None)
            if not candidates:
                raise RuntimeError("MetaDrive did not report its actual scenario index")
            if any(
                isinstance(value, bool) or not isinstance(value, Integral) for value in candidates
            ):
                raise TypeError("MetaDrive scenario index must be an integer")
            actual_values = tuple(int(value) for value in candidates)
            if len(set(actual_values)) != 1:
                raise RuntimeError(
                    "MetaDrive scenario index mismatch between reset info and runtime: "
                    f"{actual_values}"
                )
            actual = actual_values[0]
            if seed is not None and actual != seed:
                raise RuntimeError(
                    f"MetaDrive scenario index mismatch: requested {seed}, returned {actual}"
                )
            info["metadrive_scenario_index"] = actual
            return observation, info

        def setup_engine(self) -> None:
            super().setup_engine()
            self.engine.register_manager("scenario_actor_manager", ScenarioActorManager())

        def scenario_road_geometry(self) -> RoadGeometry:
            lane = self.vehicle.navigation.current_lane
            longitudinal, lateral = lane.local_coordinates(self.vehicle.position)
            lane_index = tuple(self.vehicle.lane_index)
            if len(lane_index) != 3:
                raise RuntimeError("ego lane index must contain three values")
            return RoadGeometry(
                ego_lane_index=(str(lane_index[0]), str(lane_index[1]), int(lane_index[2])),
                ego_longitudinal_m=float(longitudinal),
                ego_lateral_m=float(lateral),
                ego_speed_mps=float(self.vehicle.speed),
                decision_interval_s=float(self.config["physics_world_step_size"])
                * int(self.config["decision_repeat"]),
            )

        def scenario_spawn_lane_vehicle(self, spawn: LaneVehicleSpawn) -> str:
            return self._scenario_actor_manager().spawn_lane_vehicle(spawn)

        def scenario_spawn_crossing_actor(self, spawn: KinematicActorSpawn) -> str:
            return self._scenario_actor_manager().spawn_crossing_actor(spawn)

        def scenario_spawn_occluder(self, spawn: StaticOccluderSpawn) -> str:
            return self._scenario_actor_manager().spawn_occluder(spawn)

        def scenario_command_actor(self, actor_id: str, command: ActorCommand) -> None:
            self._scenario_actor_manager().command_actor(actor_id, command)

        def scenario_actor_state(self, actor_id: str) -> ScenarioActorState:
            return self._scenario_actor_manager().actor_state(actor_id)

        def scenario_actor_ids(self) -> tuple[str, ...]:
            return self._scenario_actor_manager().actor_ids()

        def _scenario_actor_manager(self) -> ScenarioActorManager:
            manager = self.engine.scenario_actor_manager
            if not isinstance(manager, ScenarioActorManager):
                raise RuntimeError("scenario Actor manager is not registered")
            return manager

    return cast(DrivingEnvironment, ControlMetaDriveEnv(config))
