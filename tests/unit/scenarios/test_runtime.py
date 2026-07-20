import pytest

from mad_driving.interfaces import OcclusionRegion
from mad_driving.scenarios import (
    EpisodeSeeds,
    NoOpScenarioRuntime,
    ScenarioObservationContext,
    ScenarioState,
    ScenarioStepResult,
)


class FakeEnvironment:
    pass


def test_noop_runtime_has_stable_lifecycle_outputs() -> None:
    runtime = NoOpScenarioRuntime("phase4_noop")
    seeds = EpisodeSeeds(
        episode_rng_seed=42,
        metadrive_scenario_index=7,
        scenario_parameter_seed=11,
    )
    state = runtime.reset(FakeEnvironment(), seeds=seeds)
    runtime.after_simulator_reset(FakeEnvironment(), state)
    runtime.before_step(FakeEnvironment(), state, step_index=1)
    result = runtime.after_step(
        FakeEnvironment(), state, step_index=1, raw_info={}
    )
    assert result == ScenarioStepResult(success=False, failure=False)
    assert runtime.observation_context(state) == ScenarioObservationContext(
        scenario_id="phase4_noop",
        stop_required=False,
        occlusion_regions=(),
        distance_to_conflict_point_m=None,
        intersection_entry_prohibited=False,
        visible_actor_ids=None,
    )


def test_active_occlusion_requires_visibility_metadata() -> None:
    region = OcclusionRegion(
        region_id="building-corner",
        boundary_points_xy_m=((0.0, 0.0), (1.0, 0.0)),
    )
    with pytest.raises(ValueError, match="visible_actor_ids"):
        ScenarioObservationContext(
            scenario_id="occluded",
            occlusion_regions=(region,),
            visible_actor_ids=None,
        )


def test_observation_context_freezes_visibility_and_rejects_duplicate_region_ids() -> None:
    region = OcclusionRegion(
        region_id="building-corner",
        boundary_points_xy_m=((0.0, 0.0), (1.0, 0.0)),
    )
    context = ScenarioObservationContext(
        scenario_id="occluded",
        occlusion_regions=(region,),
        visible_actor_ids=["vehicle-1", "vehicle-1"],
    )

    assert context.visible_actor_ids == frozenset({"vehicle-1"})
    with pytest.raises(ValueError, match="region_id"):
        ScenarioObservationContext(
            scenario_id="occluded",
            occlusion_regions=(region, region),
            visible_actor_ids=(),
        )


def test_scenario_state_copies_and_freezes_parameters() -> None:
    parameters = {"crossing_speed_mps": 3.0}
    state = ScenarioState(
        scenario_id="crossing",
        seeds=EpisodeSeeds(1, 2, 3),
        parameters=parameters,
    )
    parameters["crossing_speed_mps"] = 99.0

    assert state.parameters == {"crossing_speed_mps": 3.0}
    with pytest.raises(TypeError):
        state.parameters["crossing_speed_mps"] = 5.0


def test_observation_context_rejects_malformed_occlusion_regions() -> None:
    class MalformedRegion:
        region_id = "unvalidated"

    with pytest.raises(ValueError, match="occlusion_regions"):
        ScenarioObservationContext(
            scenario_id="occluded",
            occlusion_regions=(MalformedRegion(),),  # type: ignore[arg-type]
            visible_actor_ids=(),
        )


def test_observation_context_rejects_non_string_visible_actor_ids() -> None:
    with pytest.raises(ValueError, match="visible_actor_ids"):
        ScenarioObservationContext(
            scenario_id="clear",
            visible_actor_ids=("vehicle-1", 2),  # type: ignore[arg-type]
        )


def test_observation_context_rejects_bare_string_visible_actor_ids() -> None:
    with pytest.raises(ValueError, match="visible_actor_ids"):
        ScenarioObservationContext(
            scenario_id="clear",
            visible_actor_ids="vehicle-1",  # type: ignore[arg-type]
        )
