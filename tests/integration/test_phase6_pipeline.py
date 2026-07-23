from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from mad_driving.cli.evaluate import run_evaluation_bundle
from mad_driving.config.models import AppConfig
from mad_driving.evaluation.metrics import EpisodeMetricRecord, EpisodeMetrics
from mad_driving.evaluation.models import (
    EVALUATION_CASES,
    REWARD_COMPONENT_KEYS,
    EvaluationEpisodeKey,
    EvaluationEpisodeRecord,
    EvaluationRunSpec,
    Phase6PublicationPlan,
    expected_runtime_shield_mode,
)
from mad_driving.evaluation.plans import build_smoke_plan
from mad_driving.evaluation.policies import PpoPolicyAdapter, VisibleTtcRulePolicy
from mad_driving.evaluation.selection import CheckpointCandidate, CheckpointScore
from mad_driving.evaluation.training_metrics import (
    REQUIRED_TENSORBOARD_TAGS,
    TrainingMetricPoint,
)
from mad_driving.interfaces import (
    CriticReview,
    DecisionTrace,
    EgoState,
    RoadContext,
    SceneObservation,
)
from mad_driving.methods import MethodProfileSnapshot
from mad_driving.scenarios import EpisodeSeedAllocator
from mad_driving.visualization import METHOD_ORDER, PLOT_INVENTORY, _verify_bundle

_PPO_METHODS = tuple(method for method in METHOD_ORDER if method != "b0_rule")
_MINIMUM_BUNDLE = {
    "config_resolved.yaml",
    "evaluation_plan.yaml",
    "evaluation_manifest.json",
    "model_selection.csv",
    "selected_checkpoints.json",
    "episodes/proposed/system/42/level1_lead_brake/episode_20000_trace.jsonl",
    "episodes/proposed/system/42/level1_lead_brake/episode_20000_summary.json",
    "metrics/train_metrics.csv",
    "metrics/eval_metrics.csv",
    "metrics/comparison.csv",
    *(f"plots/{name}" for name in PLOT_INVENTORY),
    "renders/proposed_42_level1_lead_brake_20000.gif",
    "comparison_report.md",
}


class _FixedModel:
    def predict(
        self, observation: np.ndarray[Any, Any], **kwargs: object
    ) -> tuple[np.ndarray, None]:
        del observation, kwargs
        return np.array([0]), None


def _app_config(method_id: str) -> AppConfig:
    return AppConfig.model_validate(
        {
            "seed": 42,
            "scenario_id": "phase6-fake",
            "decision_steps": 1,
            "fixed_action": [0.0, 0.25],
            "method": {"id": method_id},
            "metadrive": {"use_render": False},
        }
    )


def _scene(*, step_index: int) -> SceneObservation:
    return SceneObservation(
        step_index=step_index,
        sim_time_s=step_index * 0.1,
        ego=EgoState((float(step_index), 0.0), 8.0, 0.0, 0.0, 0.0, 0.1, 13.0),
        visible_actors=(),
        occlusion_regions=(),
        road_context=RoadContext(False, None, False),
        previous_executed_action=0,
        previous_shield_intervention=False,
    )


class _FakeEnvironment:
    def __init__(self, spec: EvaluationRunSpec, config: AppConfig, *, fail: bool = False) -> None:
        self._spec = spec
        self._config = config
        self._fail = fail
        self._level: int | None = None
        self._scenario_id: str | None = None
        self._shield_mode: str | None = None
        self._seeds = None
        self.closed = False

    def set_difficulty_level(self, level: int) -> None:
        self._level = level

    def set_evaluation_scenario_schedule(self, scenario_ids: tuple[str, ...]) -> None:
        assert len(scenario_ids) == 1
        self._scenario_id = scenario_ids[0]

    def set_evaluation_shield_mode(self, mode: str) -> None:
        self._shield_mode = mode

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        assert seed == self._spec.test_seed
        assert options == {
            "test_seed": self._spec.test_seed,
            "scenario_cell_id": self._spec.scenario_cell_id,
        }
        self._seeds = EpisodeSeedAllocator("test", self._config.scenarios.test, 0).allocate(seed)
        return np.zeros(24, dtype=np.float32), self._identity()

    def current_scene_observation_for_evaluation(self) -> SceneObservation:
        return _scene(step_index=0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._fail:
            raise RuntimeError("injected fake simulator failure")
        assert self._seeds is not None
        components = {name: 0.0 for name in REWARD_COMPONENT_KEYS}
        components["progress_reward"] = 1.0
        profile = MethodProfileSnapshot.from_method_id(self._spec.method_id)
        trace = DecisionTrace(
            step_index=0,
            raw_action=action,
            required_action=action,
            executed_action=action,
            intervention_required=False,
            target_speed_mps=8.0,
            shield_intervened=False,
            shield_reasons=(),
            claims=(),
            review=CriticReview(0.0, False, 0.0, (), (), ()),
            reward_components=components,
            expected_agent_ids=profile.specialist_ids,
            analysis_latency_ms=0.0,
            shield_latency_ms=0.0,
            episode_rng_seed=self._seeds.episode_rng_seed,
            metadrive_scenario_index=self._seeds.metadrive_scenario_index,
            scenario_selection_seed=self._seeds.scenario_selection_seed,
            scenario_parameter_seed=self._seeds.scenario_parameter_seed,
            role="test",
            worker_index=0,
            scenario_id=self._required_scenario_id,
            difficulty_level=self._required_level,
        )
        info = self._identity()
        info.update(
            {
                "decision_trace": trace,
                "simulation_time_s": 0.1,
                "decision_interval_s": 0.1,
                "ego_speed_mps": 8.0,
                "ego_longitudinal_acceleration_mps2": 0.0,
                "route_completion": 1.0,
                "route_progress_m": 10.0,
                "lane_offset_m": 0.0,
                "collision_occurred": False,
                "collision_kind": None,
                "minimum_actual_ttc_s": None,
                "minimum_actual_stopping_margin_m": None,
                "pre_step_hard_rule_constraint": False,
                "post_step_rule_violation_event": False,
                "scenario_success": True,
                "scenario_failure": False,
                "arrived": True,
                "off_road": False,
                "unnecessary_stop_duration_s": 0.0,
                "frame_path": (
                    "episodes/proposed/system/42/level1_lead_brake/episode_20000_frames/000000.png"
                    if self._spec.episode_key
                    == EvaluationEpisodeKey(
                        "proposed", "system", "test", 42, "level1_lead_brake", 20_000
                    )
                    else None
                ),
            }
        )
        return np.ones(24, dtype=np.float32), 1.0, True, False, info

    def close(self) -> None:
        self.closed = True

    @property
    def _required_scenario_id(self) -> str:
        assert self._scenario_id is not None
        return self._scenario_id

    @property
    def _required_level(self) -> int:
        assert self._level is not None
        return self._level

    def _identity(self) -> dict[str, object]:
        assert self._seeds is not None
        assert self._shield_mode == self._spec.shield_mode
        return {
            "role": "test",
            "episode_rng_seed": self._seeds.episode_rng_seed,
            "metadrive_scenario_index": self._seeds.metadrive_scenario_index,
            "scenario_selection_seed": self._seeds.scenario_selection_seed,
            "scenario_parameter_seed": self._seeds.scenario_parameter_seed,
            "scenario_id": self._required_scenario_id,
            "difficulty_level": self._required_level,
            "scenario_parameters": {
                "difficulty_level": self._required_level,
                "scenario_id": self._required_scenario_id,
            },
        }


def _policy(
    spec: EvaluationRunSpec,
    config: AppConfig,
    candidate: CheckpointCandidate | None,
) -> VisibleTtcRulePolicy | PpoPolicyAdapter:
    if spec.method_id == "b0_rule":
        assert candidate is None
        return VisibleTtcRulePolicy()
    assert candidate is not None
    resolved = config.model_dump(mode="json")
    profile = MethodProfileSnapshot.from_method_id(spec.method_id)
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
        "checkpoint_path": str(candidate.path),
        "checkpoint_sha256": candidate.sha256,
    }
    return PpoPolicyAdapter(
        _FixedModel(),
        method_id=spec.method_id,
        checkpoint_path=str(candidate.path),
        checkpoint_sha256=candidate.sha256,
        resolved_config=resolved,
        checkpoint_metadata=metadata,
    )


def _episode_metrics() -> EpisodeMetrics:
    return EpisodeMetrics(
        collision=False,
        crossing_actor_collision=False,
        near_miss=False,
        minimum_actual_ttc_s=None,
        negative_stopping_margin=False,
        minimum_stopping_margin_m=None,
        hard_rule_violation=False,
        raw_unsafe_request_rate=0.0,
        shield_intervention_rate=0.0,
        off_road=False,
        scenario_success=True,
        final_route_completion=1.0,
        average_speed_mps=8.0,
        simulated_travel_time_s=0.1,
        unnecessary_braking_event_count=0,
        unnecessary_stop_duration_s=0.0,
        longitudinal_acceleration_rms_mps2=0.0,
        maximum_deceleration_mps2=0.0,
        longitudinal_jerk_rms_mps3=None,
        agent_disagreement_eligible_steps=0,
        agent_disagreement_count=0,
        agent_disagreement_rate=None,
        critic_challenge_eligible_steps=0,
        critic_challenge_count=0,
        critic_challenge_rate=None,
        critic_found_missed_danger_count=0,
        critic_found_missed_danger_rate=None,
        critic_false_challenge_count=0,
        critic_false_challenge_rate=None,
        agent_failure_fallback_count=0,
        decision_latency_p50_ms=0.0,
        decision_latency_p95_ms=0.0,
        decision_latency_p99_ms=0.0,
        episode_reward=1.0,
    )


def _selection_scores(
    candidates: Sequence[CheckpointCandidate],
) -> tuple[CheckpointScore, ...]:
    plan_rows = tuple((case.case_id, 10_000 + index) for index, case in enumerate(EVALUATION_CASES))
    scores: list[CheckpointScore] = []
    for candidate in candidates:
        track = "ablation" if candidate.method_id.startswith("proposed_no_") else "system"
        episodes: list[EpisodeMetricRecord] = []
        for index, case in enumerate(EVALUATION_CASES):
            seed = 10_000 + index
            episode = EvaluationEpisodeRecord(
                record_schema_version=1,
                research_contract_version=7,
                episode_key=EvaluationEpisodeKey(
                    method_id=candidate.method_id,  # type: ignore[arg-type]
                    track=track,  # type: ignore[arg-type]
                    role="validation",
                    policy_seed=candidate.policy_seed,
                    case_id=case.case_id,
                    episode_rng_seed=seed,
                ),
                method_profile=MethodProfileSnapshot.from_method_id(candidate.method_id),
                checkpoint_path=str(candidate.path),
                checkpoint_sha256=candidate.sha256,
                episode_index=index,
                is_formal=False,
                shield_mode=expected_runtime_shield_mode(track, candidate.method_id),  # type: ignore[arg-type]
                episode_rng_seed=seed,
                metadrive_scenario_index=seed,
                scenario_selection_seed=30_000 + index,
                scenario_parameter_seed=40_000 + index,
                case_id=case.case_id,
                scenario_id=case.scenario_id,
                difficulty_level=case.difficulty_level,
                sampled_scenario_parameters={
                    "difficulty_level": case.difficulty_level,
                    "scenario_id": case.scenario_id,
                },
                step_count=1,
                final_step_index=0,
                simulated_duration_s=0.1,
                cumulative_reward=1.0,
                collision_occurred=False,
                collision_kind=None,
                scenario_success=True,
                scenario_failure=False,
                arrived=True,
                off_road=False,
                terminated=True,
                truncated=False,
                complete=True,
            )
            episodes.append(EpisodeMetricRecord(episode, _episode_metrics()))
        scores.append(
            CheckpointScore(
                candidate=candidate,
                training_timestep=candidate.training_timestep,
                validation_plan_sha256="9" * 64,
                validation_plan_rows=plan_rows,
                episodes=tuple(episodes),
            )
        )
    return tuple(scores)


def _training_points(candidates: Sequence[CheckpointCandidate]) -> tuple[TrainingMetricPoint, ...]:
    points: list[TrainingMetricPoint] = []
    for candidate in candidates:
        for tag in REQUIRED_TENSORBOARD_TAGS:
            value = -0.5 if tag == "train/entropy_loss" else 1.0
            points.append(
                TrainingMetricPoint(
                    candidate.path.parents[1].name,
                    candidate.method_id,
                    candidate.policy_seed,
                    candidate.training_timestep,
                    tag,
                    value,
                )
            )
            if tag == "train/entropy_loss":
                points.append(
                    TrainingMetricPoint(
                        candidate.path.parents[1].name,
                        candidate.method_id,
                        candidate.policy_seed,
                        candidate.training_timestep,
                        "policy_entropy",
                        -value,
                    )
                )
    return tuple(points)


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 48), (24, 72, 120)).save(output, format="PNG", optimize=False)
    return output.getvalue()


def _artifact_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _scientific_hashes(root: Path) -> dict[str, str]:
    deterministic = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "evaluation_manifest.json"
        and (
            path.suffix in {".yaml", ".json", ".jsonl", ".csv", ".md", ".txt"}
            or path.name.startswith("events.out.tfevents.")
        )
    }
    return {
        relative: hashlib.sha256(
            _normalized_scientific_bytes(relative, root.joinpath(relative).read_bytes())
        ).hexdigest()
        for relative in sorted(deterministic)
    }


def _normalized_scientific_bytes(relative: str, payload: bytes) -> bytes:
    latency_fields = {
        "policy_inference_latency_ms",
        "agent_analysis_latency_ms",
        "shield_latency_ms",
        "total_decision_latency_ms",
    }
    if relative.endswith("_trace.jsonl"):
        rows = []
        for line in payload.decode("utf-8").splitlines():
            row = json.loads(line)
            for field in latency_fields:
                row[field] = 0.0
            rows.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
        return ("\n".join(rows) + "\n").encode("utf-8")
    if relative == "metrics/eval_metrics.csv":
        return _normalized_latency_csv(
            payload,
            columns={
                "decision_latency_p50_ms",
                "decision_latency_p95_ms",
                "decision_latency_p99_ms",
            },
        )
    if relative == "metrics/comparison.csv":
        source = io.StringIO(payload.decode("utf-8"), newline="")
        reader = csv.DictReader(source)
        rows = list(reader)
        assert reader.fieldnames is not None
        for row in rows:
            if row["metric"].startswith("decision_latency_"):
                row["mean"] = "0"
                row["policy_seed_stdev"] = ""
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode("utf-8")
    if relative == "comparison_report.md":
        lines: list[str] = []
        in_latency = False
        for line in payload.decode("utf-8").splitlines():
            if line == "### Latency":
                in_latency = True
            elif in_latency and line.startswith("##"):
                in_latency = False
            if in_latency and line.startswith("|") and "---" not in line:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if cells[:2] != ["Case", "Method"]:
                    cells[2:] = ["<latency>" for _ in cells[2:]]
                    line = "| " + " | ".join(cells) + " |"
            lines.append(line)
        return ("\n".join(lines) + "\n").encode("utf-8")
    return payload


def _normalized_latency_csv(payload: bytes, *, columns: set[str]) -> bytes:
    source = io.StringIO(payload.decode("utf-8"), newline="")
    reader = csv.DictReader(source)
    rows = list(reader)
    assert reader.fieldnames is not None
    for row in rows:
        for column in columns:
            row[column] = "0"
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


@pytest.fixture
def phase6_inputs(tmp_path: Path) -> Mapping[str, object]:
    training_dirs: dict[str, Path] = {}
    candidates: list[CheckpointCandidate] = []
    for index, method_id in enumerate(_PPO_METHODS, start=1):
        run_dir = tmp_path / "training" / f"{method_id}-42"
        checkpoint = run_dir / "checkpoints" / "final_model.zip"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"fake checkpoint {method_id}\n".encode())
        event = run_dir / "tensorboard" / "events.out.tfevents.fake"
        event.parent.mkdir()
        event.write_bytes(f"fake events {method_id}\n".encode())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        training_dirs[method_id] = run_dir
        candidates.append(
            CheckpointCandidate(
                path=checkpoint,
                sha256=digest,
                method_id=method_id,
                policy_seed=42,
                checkpoint_kind="final",
                curriculum_level=3,
                training_timestep=100 + index,
            )
        )
    overlays = tuple(tmp_path / "overlays" / f"{method}.yaml" for method in METHOD_ORDER)
    for overlay in overlays:
        overlay.parent.mkdir(exist_ok=True)
        overlay.write_text(f"method:\n  id: {overlay.stem}\n", encoding="utf-8")
    config_path = tmp_path / "base.yaml"
    config_path.write_text("method:\n  id: proposed\n", encoding="utf-8")
    plan = Phase6PublicationPlan.model_validate(
        {
            "plan_kind": "phase6_smoke",
            "evaluation_id": "fake-e2e",
            "is_formal": False,
            "result_label": "SMOKE - NOT A RESEARCH RESULT",
            "app_config_path": str(config_path),
            "method_overlays": [str(path) for path in overlays],
            "max_episode_steps": 1,
            "episodes_per_case": 1,
            "test_seed_start": 20_000,
            "ppo_run_bindings": [
                {
                    "method_id": candidate.method_id,
                    "policy_seed": 42,
                    "training_run_dir": str(training_dirs[candidate.method_id]),
                    "checkpoint_path": str(candidate.path),
                }
                for candidate in candidates
            ],
            "capture_episode_keys": ["proposed_system_42_level1_lead_brake_20000"],
        }
    )
    plan_path = tmp_path / "phase6_fake.yaml"
    plan_path.write_text(
        yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    checkpoint_paths = {
        (candidate.method_id, candidate.policy_seed): str(candidate.path)
        for candidate in candidates
    }
    return {
        "evaluation_config": plan,
        "plan_path": plan_path,
        "run_plan": build_smoke_plan(plan, checkpoint_paths),
        "method_configs": tuple(_app_config(method) for method in METHOD_ORDER),
        "method_profiles": tuple(
            MethodProfileSnapshot.from_method_id(method) for method in METHOD_ORDER
        ),
        "cli_overlays": (),
        "authenticated_checkpoints": tuple(candidates),
        "selection_scores": _selection_scores(candidates),
        "training_points": _training_points(candidates),
    }


def _run(
    inputs: Mapping[str, object],
    destination: Path,
    *,
    fail_after: int | None = None,
    checkpoint_reads: list[Path] | None = None,
) -> Path:
    environment_calls = 0

    def environment_factory(spec: EvaluationRunSpec, config: AppConfig) -> _FakeEnvironment:
        nonlocal environment_calls
        environment_calls += 1
        return _FakeEnvironment(spec, config, fail=environment_calls == fail_after)

    def checkpoint_reader(path: Path) -> str:
        if checkpoint_reads is not None:
            checkpoint_reads.append(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def event_reader(run_dirs: Sequence[Path], *, smoke: bool) -> tuple[TrainingMetricPoint, ...]:
        assert smoke is True
        assert run_dirs
        return inputs["training_points"]  # type: ignore[return-value]

    def frame_provider(spec: EvaluationRunSpec, record_count: int) -> tuple[bytes, ...]:
        assert spec.episode_key == EvaluationEpisodeKey(
            "proposed", "system", "test", 42, "level1_lead_brake", 20_000
        )
        return tuple(_png_bytes() for _ in range(record_count))

    return run_evaluation_bundle(
        evaluation_config=inputs["evaluation_config"],  # type: ignore[arg-type]
        plan_path=inputs["plan_path"],  # type: ignore[arg-type]
        run_plan=inputs["run_plan"],  # type: ignore[arg-type]
        method_configs=inputs["method_configs"],  # type: ignore[arg-type]
        method_profiles=inputs["method_profiles"],  # type: ignore[arg-type]
        cli_overlays=inputs["cli_overlays"],  # type: ignore[arg-type]
        authenticated_checkpoints=inputs["authenticated_checkpoints"],  # type: ignore[arg-type]
        destination=destination,
        smoke=True,
        environment_factory=environment_factory,
        policy_factory=_policy,
        checkpoint_reader=checkpoint_reader,
        event_reader=event_reader,
        frame_provider=frame_provider,
        selection_scores=inputs["selection_scores"],  # type: ignore[arg-type]
    )


def test_fake_phase6_pipeline_is_strict_repeatable_and_failure_atomic(
    tmp_path: Path, phase6_inputs: Mapping[str, object]
) -> None:
    first = _run(phase6_inputs, tmp_path / "published-first")
    second = _run(phase6_inputs, tmp_path / "published-second")

    assert _MINIMUM_BUNDLE <= _artifact_files(first)
    assert _MINIMUM_BUNDLE <= _artifact_files(second)
    assert _scientific_hashes(first) == _scientific_hashes(second)
    for relative in (*_MINIMUM_BUNDLE,):
        if relative.endswith((".png", ".gif")):
            assert first.joinpath(relative).read_bytes() == second.joinpath(relative).read_bytes()

    manifest = json.loads(first.joinpath("evaluation_manifest.json").read_bytes())
    inventory = {item["path"]: item for item in manifest["artifacts"]}
    assert set(inventory) == _artifact_files(first) - {"evaluation_manifest.json"}
    event_paths = sorted(path for path in inventory if "/tensorboard/" in path)
    assert len(event_paths) == len(_PPO_METHODS)
    for relative, item in inventory.items():
        payload = first.joinpath(relative).read_bytes()
        assert item == {
            "path": relative,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    first.joinpath("metrics", "comparison.csv").write_bytes(b"corrupt\n")
    with pytest.raises(ValueError, match="SHA-256|manifest"):
        _verify_bundle(first)

    failed = tmp_path / "failed-publication"
    with pytest.raises(RuntimeError, match="injected fake simulator failure"):
        _run(phase6_inputs, failed, fail_after=7)
    assert not failed.exists()
    assert not tuple(tmp_path.glob(".failed-publication.*"))

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    reads: list[Path] = []
    before = set(tmp_path.iterdir())
    with pytest.raises(FileExistsError):
        _run(phase6_inputs, occupied, checkpoint_reads=reads)
    assert reads == []
    assert set(tmp_path.iterdir()) == before
