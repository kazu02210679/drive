from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mad_driving.config.models import AppConfig
from mad_driving.control import DrivingAction
from mad_driving.evaluation.models import REWARD_COMPONENT_KEYS, EvaluationRunSpec
from mad_driving.evaluation.policies import PpoPolicyAdapter, VisibleTtcRulePolicy
from mad_driving.evaluation.runner import run_evaluation_episode
from mad_driving.interfaces import (
    CriticReview,
    DecisionTrace,
    EgoState,
    RoadContext,
    SceneObservation,
)
from mad_driving.methods import MethodProfileSnapshot
from mad_driving.scenarios import EpisodeSeedAllocator


def config(method_id: str = "b0_rule") -> AppConfig:
    return AppConfig.model_validate(
        {
            "seed": 42,
            "scenario_id": "phase5",
            "decision_steps": 8,
            "fixed_action": [0.0, 0.25],
            "method": {"id": method_id},
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


def ppo_spec(*, track: str, method_id: str, shield_mode: str) -> EvaluationRunSpec:
    return EvaluationRunSpec(
        track=track,  # type: ignore[arg-type]
        method_id=method_id,  # type: ignore[arg-type]
        policy_seed=42,
        checkpoint_path="runs/model.zip",
        scenario_cell_id="level1_lead_brake",
        episode_index=0,
        test_seed=20_001,
        shield_mode=shield_mode,  # type: ignore[arg-type]
        is_formal=False,
    )


class FixedPpoModel:
    def __init__(self) -> None:
        self.predict_calls = 0
        self.observation_shapes: list[tuple[int, ...]] = []

    def predict(self, observation: np.ndarray, **kwargs: object) -> tuple[np.ndarray, None]:
        del kwargs
        self.predict_calls += 1
        self.observation_shapes.append(observation.shape)
        return np.array([0]), None


def ppo_policy(
    method_id: str, selected_config: AppConfig, model: FixedPpoModel | None = None
) -> PpoPolicyAdapter:
    path = "runs/model.zip"
    digest = "a" * 64
    resolved = selected_config.model_dump(mode="json")
    profile = MethodProfileSnapshot.from_method_id(selected_config.method.id)
    metadata = {
        "research_contract_version": 7,
        "observation_schema_version": 1,
        "observation_shape": (24,),
        "observation_dtype": "float32",
        "action_schema_version": 1,
        "action_count": 4,
        "action_order": ("KEEP", "SLOW", "PREPARE_STOP", "STOP"),
        "method_profile": profile,
        "resolved_config": resolved,
        "checkpoint_path": path,
        "checkpoint_sha256": digest,
    }
    return PpoPolicyAdapter(
        model or FixedPpoModel(),
        method_id=method_id,  # type: ignore[arg-type]
        checkpoint_path=path,
        checkpoint_sha256=digest,
        resolved_config=resolved,
        checkpoint_metadata=metadata,
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
    def __init__(
        self,
        *,
        fail_step: bool = False,
        truncate: bool = False,
        expected_agent_ids: tuple[str, ...] = (),
        step_identity_drift: tuple[str, str, object] | None = None,
    ) -> None:
        self.fail_step = fail_step
        self.truncate = truncate
        self.levels: list[int] = []
        self.schedules: list[tuple[str, ...]] = []
        self.shield_modes: list[str] = []
        self.reset_calls: list[tuple[int | None, dict[str, Any] | None]] = []
        self.actions: list[int] = []
        self.close_calls = 0
        self.expected_agent_ids = expected_agent_ids
        self.step_identity_drift = step_identity_drift
        self.scene = visible_scene(0, speed_limit_mps=15.0)
        self.seeds = EpisodeSeedAllocator("test", config().scenarios.test, 0).allocate(20_001)

    def set_difficulty_level(self, level: int) -> None:
        self.levels.append(level)

    def set_evaluation_scenario_schedule(self, scenario_ids: tuple[str, ...]) -> None:
        self.schedules.append(scenario_ids)

    def set_evaluation_shield_mode(self, mode: str) -> None:
        self.shield_modes.append(mode)

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
            expected_agent_ids=self.expected_agent_ids,
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
        if self.step_identity_drift is not None and self.step_identity_drift[0] == "trace":
            _, field, value = self.step_identity_drift
            trace = replace(trace, **{field: value})
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
        if self.step_identity_drift is not None and self.step_identity_drift[0] == "info":
            _, field, value = self.step_identity_drift
            info[field] = value
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
            "role": "test",
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
    assert environment.shield_modes == ["enforce"]
    assert environment.reset_calls == [
        (20_001, {"test_seed": 20_001, "scenario_cell_id": "level1_lead_brake"})
    ]
    assert environment.actions == [int(DrivingAction.KEEP), int(DrivingAction.KEEP)]
    assert environment.close_calls == 1
    assert [record.step_index for record in result.step_records] == [0, 1]
    assert {record.episode_index for record in result.step_records} == {0}
    assert {record.is_formal for record in result.step_records} == {False}
    assert [record.ego_speed_limit_mps for record in result.step_records] == [15.0, 14.0]
    assert result.episode_record.episode_index == 0
    assert result.episode_record.is_formal is False
    assert result.episode_record.step_count == 2
    assert result.episode_record.cumulative_reward == 2.0
    assert getattr(result.episode_record, terminal_field) is True
    assert destination.joinpath("steps.jsonl").is_file()
    assert destination.joinpath("episode.jsonl").is_file()
    assert destination.joinpath("evaluation_manifest.json").is_file()


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


@pytest.mark.parametrize(
    ("track", "method_id", "shield_mode"),
    [
        ("decision", "proposed", "monitor"),
        ("system", "b0_rule", "enforce"),
        ("ablation", "proposed_no_shield", "off"),
    ],
)
def test_runner_installs_the_fixed_track_shield_mode_before_reset(
    tmp_path: Path, track: str, method_id: str, shield_mode: str
) -> None:
    selected_spec = (
        spec()
        if method_id == "b0_rule"
        else ppo_spec(track=track, method_id=method_id, shield_mode=shield_mode)
    )
    selected_config = config(method_id)
    policy = (
        VisibleTtcRulePolicy() if method_id == "b0_rule" else ppo_policy(method_id, selected_config)
    )
    environment = FakeEvaluationEnv(
        expected_agent_ids=MethodProfileSnapshot.from_method_id(
            selected_config.method.id
        ).specialist_ids
    )

    run_evaluation_episode(
        selected_spec,
        environment=environment,
        policy=policy,
        config=selected_config,
        destination=tmp_path / method_id,
        checkpoint_sha256=None if method_id == "b0_rule" else "a" * 64,
    )

    assert environment.shield_modes == [shield_mode]
    assert environment.reset_calls


def test_runner_preserves_24d_ppo_observation_and_explicit_provenance(tmp_path: Path) -> None:
    selected_config = config("proposed")
    model = FixedPpoModel()
    policy = ppo_policy("proposed", selected_config, model)
    selected_spec = replace(
        ppo_spec(track="system", method_id="proposed", shield_mode="enforce"),
        episode_index=4,
        is_formal=True,
    )

    result = run_evaluation_episode(
        selected_spec,
        environment=FakeEvaluationEnv(expected_agent_ids=("nominal", "hazard", "rule")),
        policy=policy,
        config=selected_config,
        destination=tmp_path / "evaluation",
        checkpoint_sha256="a" * 64,
    )

    assert model.observation_shapes == [(24,), (24,)]
    assert {(record.episode_index, record.is_formal) for record in result.step_records} == {
        (4, True)
    }
    assert (result.episode_record.episode_index, result.episode_record.is_formal) == (4, True)


class CountingRulePolicy(VisibleTtcRulePolicy):
    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, observation: SceneObservation | np.ndarray[Any, Any]) -> int:
        self.predict_calls += 1
        return super().predict(observation)


@pytest.mark.parametrize(
    "malformed_spec",
    [
        EvaluationRunSpec(
            track="system",
            method_id="b0_rule",
            policy_seed=None,
            checkpoint_path=None,
            scenario_cell_id="level1_lead_brake",
            episode_index=0,
            test_seed=20_001,
            shield_mode="monitor",
            is_formal=False,
        ),
        EvaluationRunSpec(
            track="decision",
            method_id="b0_rule",
            policy_seed=None,
            checkpoint_path=None,
            scenario_cell_id="level1_lead_brake",
            episode_index=0,
            test_seed=20_001,
            shield_mode="monitor",
            is_formal=False,
        ),
    ],
)
def test_runner_rejects_malformed_track_shield_combinations_before_prediction(
    tmp_path: Path, malformed_spec: EvaluationRunSpec
) -> None:
    environment = FakeEvaluationEnv()
    policy = CountingRulePolicy()

    with pytest.raises(ValueError, match="matrix|shield"):
        run_evaluation_episode(
            malformed_spec,
            environment=environment,
            policy=policy,
            config=config(),
            destination=tmp_path / "evaluation",
        )

    assert policy.predict_calls == 0
    assert not environment.reset_calls
    assert not environment.shield_modes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_path", "   "),
        ("checkpoint_sha256", ""),
        ("checkpoint_sha256", "A" * 64),
    ],
)
def test_runner_revalidates_malformed_checkpoint_binding_before_environment_setup(
    tmp_path: Path, field: str, value: str
) -> None:
    selected_config = config("proposed")
    model = FixedPpoModel()
    policy = ppo_policy("proposed", selected_config, model)
    selected_spec = ppo_spec(track="system", method_id="proposed", shield_mode="enforce")
    if field == "checkpoint_path":
        policy.checkpoint_path = value
        selected_spec = replace(selected_spec, checkpoint_path=value)
        digest = "a" * 64
    else:
        policy.checkpoint_sha256 = value
        digest = value
    environment = FakeEvaluationEnv(expected_agent_ids=("nominal", "hazard", "rule"))

    with pytest.raises(ValueError, match="checkpoint"):
        run_evaluation_episode(
            selected_spec,
            environment=environment,
            policy=policy,
            config=selected_config,
            destination=tmp_path / "evaluation",
            checkpoint_sha256=digest,
        )

    assert model.predict_calls == 0
    assert not environment.reset_calls
    assert not environment.shield_modes
    assert not environment.levels
    assert not environment.schedules


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


@pytest.mark.parametrize("carrier", ["info", "trace"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "validation"),
        ("episode_rng_seed", 20_002),
        ("metadrive_scenario_index", 20_002),
        ("scenario_selection_seed", 1),
        ("scenario_parameter_seed", 1),
        ("scenario_id", "cut_in"),
        ("difficulty_level", 2),
    ],
)
def test_runner_rejects_each_transition_identity_drift_before_writing(
    tmp_path: Path, carrier: str, field: str, value: object
) -> None:
    environment = FakeEvaluationEnv(step_identity_drift=(carrier, field, value))
    destination = tmp_path / "evaluation"

    with pytest.raises(RuntimeError, match=f"step.*{field}.*mismatch"):
        run_evaluation_episode(
            spec(),
            environment=environment,
            policy=VisibleTtcRulePolicy(),
            config=config(),
            destination=destination,
        )

    assert not destination.exists()
    staging = next(tmp_path.glob(".evaluation.staging-*"))
    assert staging.joinpath("steps.jsonl").read_bytes() == b""


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
