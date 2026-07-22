"""Lazy scenario API that keeps metadata-only consumers simulator-free."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mad_driving.scenarios.actor_manager import ScenarioActorManager
    from mad_driving.scenarios.actors import (
        ActorCommand,
        KinematicActorSpawn,
        LanePoseCommand,
        LaneVehicleSpawn,
        RoadGeometry,
        ScenarioActorCommand,
        ScenarioActorState,
        StaticOccluderSpawn,
        VelocityCommand,
    )
    from mad_driving.scenarios.cut_in import CutInRuntime
    from mad_driving.scenarios.factory import (
        ScenarioRuntimeFactory,
        build_scenario_runtime_factory,
    )
    from mad_driving.scenarios.lead_brake import LeadBrakeRuntime, NominalScenarioRuntime
    from mad_driving.scenarios.manager import ScenarioManagerRuntime
    from mad_driving.scenarios.occluded_crossing import OccludedCrossingRuntime
    from mad_driving.scenarios.parameters import ScenarioParameterSampler
    from mad_driving.scenarios.runtime import (
        NoOpScenarioRuntime,
        ScenarioObservationContext,
        ScenarioRuntime,
        ScenarioState,
        ScenarioStepResult,
        ScenarioTransition,
    )
    from mad_driving.scenarios.seeding import (
        EnvironmentRole,
        EpisodeSeedAllocator,
        EpisodeSeeds,
    )

_EXPORT_MODULES = {
    "ActorCommand": "mad_driving.scenarios.actors",
    "CutInRuntime": "mad_driving.scenarios.cut_in",
    "EnvironmentRole": "mad_driving.scenarios.seeding",
    "EpisodeSeedAllocator": "mad_driving.scenarios.seeding",
    "EpisodeSeeds": "mad_driving.scenarios.seeding",
    "KinematicActorSpawn": "mad_driving.scenarios.actors",
    "LanePoseCommand": "mad_driving.scenarios.actors",
    "LaneVehicleSpawn": "mad_driving.scenarios.actors",
    "LeadBrakeRuntime": "mad_driving.scenarios.lead_brake",
    "NoOpScenarioRuntime": "mad_driving.scenarios.runtime",
    "NominalScenarioRuntime": "mad_driving.scenarios.lead_brake",
    "OccludedCrossingRuntime": "mad_driving.scenarios.occluded_crossing",
    "RoadGeometry": "mad_driving.scenarios.actors",
    "ScenarioActorCommand": "mad_driving.scenarios.actors",
    "ScenarioActorManager": "mad_driving.scenarios.actor_manager",
    "ScenarioActorState": "mad_driving.scenarios.actors",
    "ScenarioManagerRuntime": "mad_driving.scenarios.manager",
    "ScenarioObservationContext": "mad_driving.scenarios.runtime",
    "ScenarioParameterSampler": "mad_driving.scenarios.parameters",
    "ScenarioRuntime": "mad_driving.scenarios.runtime",
    "ScenarioRuntimeFactory": "mad_driving.scenarios.factory",
    "ScenarioState": "mad_driving.scenarios.runtime",
    "ScenarioStepResult": "mad_driving.scenarios.runtime",
    "ScenarioTransition": "mad_driving.scenarios.runtime",
    "StaticOccluderSpawn": "mad_driving.scenarios.actors",
    "VelocityCommand": "mad_driving.scenarios.actors",
    "build_scenario_runtime_factory": "mad_driving.scenarios.factory",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORT_MODULES))


__all__ = [
    "ActorCommand",
    "CutInRuntime",
    "EnvironmentRole",
    "EpisodeSeedAllocator",
    "EpisodeSeeds",
    "KinematicActorSpawn",
    "LanePoseCommand",
    "LaneVehicleSpawn",
    "LeadBrakeRuntime",
    "NoOpScenarioRuntime",
    "NominalScenarioRuntime",
    "OccludedCrossingRuntime",
    "RoadGeometry",
    "ScenarioActorCommand",
    "ScenarioActorManager",
    "ScenarioActorState",
    "ScenarioManagerRuntime",
    "ScenarioObservationContext",
    "ScenarioParameterSampler",
    "ScenarioRuntime",
    "ScenarioRuntimeFactory",
    "ScenarioState",
    "ScenarioStepResult",
    "ScenarioTransition",
    "StaticOccluderSpawn",
    "VelocityCommand",
    "build_scenario_runtime_factory",
]
