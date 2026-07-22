from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mad_driving.config.models import AppConfig
from mad_driving.control import DrivingAction
from mad_driving.evaluation.models import REWARD_COMPONENT_KEYS, EvaluationRunSpec
from mad_driving.evaluation.policies import VisibleTtcRulePolicy
from mad_driving.evaluation.runner import run_evaluation_episode
from mad_driving.interfaces import (
    CriticReview,
    DecisionTrace,
    EgoState,
    RoadContext,
    SceneObservation,
)
from mad_driving.scenarios import EpisodeSeedAllocator


def config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "seed": 42,
            "scenario_id": "phase5",
            "decision_steps": 8,
            "fixed_action": [0.0, 0.25],
            "method": {"id": "b0_rule"},
            "metadrive": {"use_render": False},
        }
    )


def spec() -> EvaluationRunSpec:
    return EvaluationRunSpec(
        track="system",
        method_id="b0_rule",
        policy_seed=None,
        checkpoint_path=None,
        scenario_cell_id="level1_lead_brake",
        episode_index=0,
        test_seed=20_001,
        shield_mode="enforce",
        is_formal=False,
    )


def visible_scene(step_index: int, *, speed_limit_mps: float) -> SceneObservation:
    return SceneObservation(
        step_index=step_index,
        sim_time_s=step_index * 0.1,
        ego=EgoState((float(step_index), 0.0), 10.0, 0.0, 0.0, 0.0, 0.1, speed_limit_mps),
        visible_actors=(),
        occlusion_regions=(),
        road_context=RoadContext(False, None, False),
        previous_executed_action=int(DrivingAction.KEEP),
        previous_shield_intervention=False,
    )


class FakeEvaluationEnv:
    def __init__(self, *, fail_step: bool = False, truncate: bool = False) -> None:
        self.fail_step = fail_step
        self.truncate = truncate
        self.levels: list[int] = []
        self.schedules: list[tuple[str, ...]] = []
        self.reset_calls: list[tuple[int | None, dict[str, Any] | None]] = []
        self.actions: list[int] = []
        self.close_calls = 0
        self.scene = visible_scene(0, speed_limit_mps=15.0)
        self.seeds = EpisodeSeedAllocator("test", config().scenarios.test, 0).allocate(20_001)

    def set_difficulty_level(self, level: int) -> None:
        self.levels.append(level)

    def set_evaluation_scenario_schedule(self, scenario_ids: tuple[str, ...]) -> None:
        self.schedules.append(scenario_ids)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        self.reset_calls.append((seed, options))
        self.scene = visible_scene(0, speed_limit_mps=15.0)
        return np.zeros(24, dtype=np.float32), self._identity_info()

    def current_scene_observation_for_evaluation(self) -> SceneObservation:
        return self.scene

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self.fail_step:
            raise RuntimeError("simulator failed")
        self.actions.append(action)
        index = len(self.actions)
        terminated = index == 2 and not self.truncate
        truncated = index == 2 and self.truncate
        self.scene = visible_scene(index, speed_limit_mps=15.0 - index)
        info = self._identity_info()
        components = {name: 0.0 for name in REWARD_COMPONENT_KEYS}
        components["progress_reward"] = 1.0
        trace = DecisionTrace(
            step_index=index,
            raw_action=action,
            required_action=action,
            executed_action=action,
            intervention_required=False,
            target_speed_mps=10.0,
            shield_intervened=False,
            shield_reasons=(),
            claims=(),
            review=CriticReview(0.0, False, 0.0, (), (), ()),
            reward_components=components,
            expected_agent_ids=(),
            analysis_latency_ms=0.2,
            shield_latency_ms=0.1,
            episode_rng_seed=self.seeds.episode_rng_seed,
            metadrive_scenario_index=self.seeds.metadrive_scenario_index,
            scenario_selection_seed=self.seeds.scenario_selection_seed,
            scenario_parameter_seed=self.seeds.scenario_parameter_seed,
            role="test",
            worker_index=0,
            scenario_id="lead_brake",
            difficulty_level=1,
        )
        info.update(
            {
                "decision_trace": trace,
                "simulation_time_s": index * 0.1,
                "decision_interval_s": 0.1,
                "ego_speed_mps": 10.0,
                "ego_longitudinal_acceleration_mps2": 0.0,
                "route_completion": index / 10.0,
                "route_progress_m": float(index),
                "lane_offset_m": 0.0,
                "collision_occurred": False,
                "collision_kind": None,
                "minimum_actual_ttc_s": None,
                "minimum_actual_stopping_margin_m": None,
                "pre_step_hard_rule_constraint": False,
                "post_step_rule_violation_event": False,
                "scenario_success": terminated,
                "scenario_failure": False,
                "arrived": False,
                "off_road": False,
                "unnecessary_stop_duration_s": 0.0,
            }
        )
        return np.full(24, index, dtype=np.float32), 1.0, terminated, truncated, info

    def close(self) -> None:
        self.close_calls += 1

    def _identity_info(self) -> dict[str, object]:
        return {
            "episode_rng_seed": self.seeds.episode_rng_seed,
            "metadrive_scenario_index": self.seeds.metadrive_scenario_index,
            "scenario_selection_seed": self.seeds.scenario_selection_seed,
            "scenario_parameter_seed": self.seeds.scenario_parameter_seed,
            "scenario_id": "lead_brake",
            "difficulty_level": 1,
            "scenario_parameters": {"difficulty_level": 1, "initial_gap_m": 35.0},
        }


@pytest.mark.parametrize(
    ("truncate", "terminal_field"), [(False, "terminated"), (True, "truncated")]
)
def test_runner_installs_plan_logs_each_step_and_derives_terminal_episode(
    tmp_path: Path, truncate: bool, terminal_field: str
) -> None:
    environment = FakeEvaluationEnv(truncate=truncate)
    destination = tmp_path / "evaluation"

    result = run_evaluation_episode(
        spec(),
        environment=environment,
        policy=VisibleTtcRulePolicy(),
        config=config(),
        destination=destination,
    )

    assert environment.levels == [1]
    assert environment.schedules == [("lead_brake",)]
    assert environment.reset_calls == [
        (20_001, {"test_seed": 20_001, "scenario_cell_id": "level1_lead_brake"})
    ]
    assert environment.actions == [int(DrivingAction.KEEP), int(DrivingAction.KEEP)]
    assert environment.close_calls == 1
    assert [record.step_index for record in result.step_records] == [0, 1]
    assert [record.ego_speed_limit_mps for record in result.step_records] == [15.0, 14.0]
    assert result.episode_record.step_count == 2
    assert result.episode_record.cumulative_reward == 2.0
    assert getattr(result.episode_record, terminal_field) is True
    assert destination.joinpath("steps.jsonl").is_file()
    assert destination.joinpath("episode.jsonl").is_file()


def test_runner_repeated_scientific_records_are_identical_without_latency(tmp_path: Path) -> None:
    destinations = (tmp_path / "first", tmp_path / "second")
    results = tuple(
        run_evaluation_episode(
            spec(),
            environment=FakeEvaluationEnv(),
            policy=VisibleTtcRulePolicy(),
            config=config(),
            destination=destination,
        )
        for destination in destinations
    )

    def scientific_bytes(result_index: int) -> bytes:
        rows = [record.to_dict() for record in results[result_index].step_records]
        for row in rows:
            for field in (
                "policy_inference_latency_ms",
                "agent_analysis_latency_ms",
                "shield_latency_ms",
                "total_decision_latency_ms",
            ):
                row.pop(field)
        return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()

    assert scientific_bytes(0) == scientific_bytes(1)
    assert results[0].episode_record == results[1].episode_record


def test_runner_failure_closes_environment_and_never_publishes(tmp_path: Path) -> None:
    environment = FakeEvaluationEnv(fail_step=True)
    destination = tmp_path / "evaluation"

    with pytest.raises(RuntimeError, match="simulator failed"):
        run_evaluation_episode(
            spec(),
            environment=environment,
            policy=VisibleTtcRulePolicy(),
            config=config(),
            destination=destination,
        )

    assert environment.close_calls == 1
    assert not destination.exists()
    assert len(tuple(tmp_path.glob(".evaluation.staging-*"))) == 1


def test_runner_existing_destination_is_rejected_and_environment_is_closed(tmp_path: Path) -> None:
    environment = FakeEvaluationEnv()
    destination = tmp_path / "evaluation"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        run_evaluation_episode(
            spec(),
            environment=environment,
            policy=VisibleTtcRulePolicy(),
            config=config(),
            destination=destination,
        )

    assert environment.close_calls == 1
