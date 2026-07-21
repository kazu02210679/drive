"""Scenario lifecycle contracts independent from the simulator integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from mad_driving.interfaces import OcclusionRegion
from mad_driving.interfaces._validation import require_finite, require_non_empty
from mad_driving.scenarios.seeding import EpisodeSeeds

if TYPE_CHECKING:
    from mad_driving.envs.multi_agent_speed_env import DrivingEnvironment


def _freeze_parameter(value: object) -> object:
    """Copy JSON-like scenario parameters into recursively immutable containers."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        require_finite("scenario parameter", value)
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ValueError("scenario parameter mapping keys must be strings")
            frozen[key] = _freeze_parameter(nested_value)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_freeze_parameter(item) for item in value)
    raise ValueError("scenario parameters must contain only finite JSON-like values")


@dataclass(frozen=True)
class ScenarioState:
    """Immutable scenario identity and parameters for one episode."""

    scenario_id: str
    seeds: EpisodeSeeds
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        require_non_empty("scenario_id", self.scenario_id)
        frozen_parameters = _freeze_parameter(self.parameters)
        if not isinstance(frozen_parameters, Mapping):
            raise ValueError("parameters must be a mapping")
        object.__setattr__(self, "parameters", frozen_parameters)


@dataclass(frozen=True)
class ScenarioStepResult:
    """Scenario outcome flags evaluated after a simulator step."""

    success: bool
    failure: bool


@dataclass(frozen=True)
class ScenarioTransition:
    """Immutable scenario state and outcome produced by one simulator step."""

    state: ScenarioState
    outcome: ScenarioStepResult


@dataclass(frozen=True)
class ScenarioObservationContext:
    """Validated scenario metadata made available to observation construction."""

    scenario_id: str
    stop_required: bool = False
    occlusion_regions: tuple[OcclusionRegion, ...] = ()
    distance_to_conflict_point_m: float | None = None
    intersection_entry_prohibited: bool = False
    visible_actor_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        require_non_empty("scenario_id", self.scenario_id)
        regions = tuple(self.occlusion_regions)
        if not all(isinstance(region, OcclusionRegion) for region in regions):
            raise ValueError("occlusion_regions must contain only OcclusionRegion values")
        region_ids = tuple(region.region_id for region in regions)
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("occlusion region_id values must be unique")
        if self.distance_to_conflict_point_m is not None:
            require_finite("distance_to_conflict_point_m", self.distance_to_conflict_point_m)
        if regions and self.visible_actor_ids is None:
            raise ValueError("visible_actor_ids are required when occlusion_regions are active")
        if isinstance(self.visible_actor_ids, str):
            raise ValueError("visible_actor_ids must not be a bare string")
        visible_actor_ids = None
        if self.visible_actor_ids is not None:
            visible_actor_ids = frozenset(self.visible_actor_ids)
            if not all(isinstance(actor_id, str) for actor_id in visible_actor_ids):
                raise ValueError("visible_actor_ids must contain only strings")
        object.__setattr__(self, "occlusion_regions", regions)
        object.__setattr__(self, "visible_actor_ids", visible_actor_ids)


class ScenarioRuntime(Protocol):
    """Lifecycle hooks that a scenario uses around simulator execution."""

    def reset(self, environment: DrivingEnvironment, *, seeds: EpisodeSeeds) -> ScenarioState: ...

    def after_simulator_reset(
        self, environment: DrivingEnvironment, state: ScenarioState
    ) -> ScenarioState: ...

    def before_step(
        self, environment: DrivingEnvironment, state: ScenarioState, *, step_index: int
    ) -> ScenarioState: ...

    def after_step(
        self,
        environment: DrivingEnvironment,
        state: ScenarioState,
        *,
        step_index: int,
        raw_info: Mapping[str, object],
    ) -> ScenarioTransition: ...

    def observation_context(self, state: ScenarioState) -> ScenarioObservationContext: ...


@dataclass(frozen=True)
class NoOpScenarioRuntime:
    """Deterministic lifecycle implementation with no simulator side effects."""

    scenario_id: str

    def __post_init__(self) -> None:
        require_non_empty("scenario_id", self.scenario_id)

    def reset(self, environment: DrivingEnvironment, *, seeds: EpisodeSeeds) -> ScenarioState:
        return ScenarioState(
            scenario_id=self.scenario_id,
            seeds=seeds,
            parameters={},
        )

    def after_simulator_reset(
        self, environment: DrivingEnvironment, state: ScenarioState
    ) -> ScenarioState:
        return state

    def before_step(
        self, environment: DrivingEnvironment, state: ScenarioState, *, step_index: int
    ) -> ScenarioState:
        return state

    def after_step(
        self,
        environment: DrivingEnvironment,
        state: ScenarioState,
        *,
        step_index: int,
        raw_info: Mapping[str, object],
    ) -> ScenarioTransition:
        return ScenarioTransition(
            state=state,
            outcome=ScenarioStepResult(success=False, failure=False),
        )

    def observation_context(self, state: ScenarioState) -> ScenarioObservationContext:
        return ScenarioObservationContext(scenario_id=state.scenario_id)
