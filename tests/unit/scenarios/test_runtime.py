import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from math import inf
from pathlib import Path

import pytest

from mad_driving.interfaces import OcclusionRegion
from mad_driving.scenarios import (
    EpisodeSeeds,
    NoOpScenarioRuntime,
    ScenarioObservationContext,
    ScenarioState,
    ScenarioStepResult,
    ScenarioTransition,
)


class FakeEnvironment:
    pass


def test_scenarios_package_imports_in_a_fresh_process() -> None:
    source_root = Path(__file__).parents[3] / "src"
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(source_root), os.environ.get("PYTHONPATH")))
        )
    }

    result = subprocess.run(
        [sys.executable, "-c", "import mad_driving.scenarios"],
        cwd=source_root.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_noop_runtime_has_stable_lifecycle_outputs() -> None:
    runtime = NoOpScenarioRuntime("phase4_noop")
    seeds = EpisodeSeeds(
        episode_rng_seed=42,
        metadrive_scenario_index=7,
        scenario_parameter_seed=11,
    )
    state = runtime.reset(FakeEnvironment(), seeds=seeds)
    reset_state = runtime.after_simulator_reset(FakeEnvironment(), state)
    step_state = runtime.before_step(FakeEnvironment(), reset_state, step_index=1)
    transition = runtime.after_step(FakeEnvironment(), step_state, step_index=1, raw_info={})
    assert reset_state is state
    assert step_state is state
    assert transition == ScenarioTransition(
        state=state,
        outcome=ScenarioStepResult(success=False, failure=False),
    )
    assert runtime.observation_context(transition.state) == ScenarioObservationContext(
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


def test_scenario_state_recursively_freezes_nested_parameter_aliases() -> None:
    levels = [1.0, 2.0]
    schedule = {"levels": levels}
    parameters = {"curriculum": [schedule]}
    state = ScenarioState(
        scenario_id="crossing",
        seeds=EpisodeSeeds(1, 2, 3),
        parameters=parameters,
    )

    levels.append(99.0)
    schedule["levels"] = [100.0]
    parameters["curriculum"].append({"levels": [200.0]})

    curriculum = state.parameters["curriculum"]
    assert isinstance(curriculum, tuple)
    frozen_schedule = curriculum[0]
    assert isinstance(frozen_schedule, Mapping)
    assert frozen_schedule["levels"] == (1.0, 2.0)
    with pytest.raises(TypeError):
        frozen_schedule["levels"] = ()  # type: ignore[index]


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({1: "value"}, "mapping keys"),
        ({"value": inf}, "finite"),
        ({"value": {1, 2}}, "JSON-like"),
        ([], "must be a mapping"),
    ],
)
def test_scenario_state_rejects_non_json_or_non_mapping_parameters(
    parameters: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ScenarioState(
            scenario_id="crossing",
            seeds=EpisodeSeeds(1, 2, 3),
            parameters=parameters,  # type: ignore[arg-type]
        )


def test_scenario_transition_is_immutable() -> None:
    state = ScenarioState("crossing", EpisodeSeeds(1, 2, 3), {})
    transition = ScenarioTransition(
        state=state,
        outcome=ScenarioStepResult(success=False, failure=False),
    )

    with pytest.raises(FrozenInstanceError):
        transition.state = ScenarioState("crossing", state.seeds, {})  # type: ignore[misc]


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
