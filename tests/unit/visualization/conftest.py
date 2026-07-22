from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from mad_driving.evaluation.compare import (
    COMPARISON_CSV_COLUMNS,
    EVAL_METRICS_CSV_COLUMNS,
)
from mad_driving.evaluation.models import (
    REWARD_COMPONENT_KEYS,
    EvaluationEpisodeKey,
    EvaluationStepRecord,
)
from mad_driving.evaluation.serialization import write_jsonl_strict
from mad_driving.evaluation.training_metrics import TRAIN_METRICS_CSV_COLUMNS
from mad_driving.evaluation.workspace import EvaluationWorkspace
from mad_driving.interfaces import CriticReview, RiskClaim
from mad_driving.methods import MethodProfileSnapshot

SMOKE = "SMOKE - NOT A RESEARCH RESULT"


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _claim(agent_id: str, severity: float, recommended_speed: float) -> RiskClaim:
    return RiskClaim(
        claim_id=f"{agent_id}:aggregate",
        agent_id=agent_id,
        event_type="aggregate",
        target_actor_id="lead",
        probability=severity,
        confidence=0.9,
        severity=severity,
        time_horizon_s=3.0,
        min_ttc_s=2.5,
        stopping_margin_m=4.0,
        recommended_max_speed_mps=recommended_speed,
        hard_stop_required=False,
        evidence=("persisted evidence",),
        assumptions=("persisted assumption",),
        valid_until_step=3,
    )


def make_step(*, step_index: int, frame_path: str) -> EvaluationStepRecord:
    components = {name: 0.0 for name in REWARD_COMPONENT_KEYS}
    components["progress_reward"] = 0.5
    claims = (
        _claim("nominal", 0.2, 12.0),
        _claim("hazard", 0.8, 4.0),
        _claim("rule", 0.5, 6.0),
    )
    return EvaluationStepRecord(
        record_schema_version=1,
        research_contract_version=7,
        episode_key=EvaluationEpisodeKey(
            method_id="proposed",
            track="system",
            role="test",
            policy_seed=42,
            case_id="level1_lead_brake",
            episode_rng_seed=20_001,
        ),
        method_profile=MethodProfileSnapshot.from_method_id("proposed"),
        checkpoint_path="runs/proposed/42.zip",
        checkpoint_sha256="a" * 64,
        episode_index=0,
        is_formal=False,
        shield_mode="enforce",
        step_index=step_index,
        simulation_time_s=step_index * 0.1,
        decision_interval_s=0.1,
        episode_rng_seed=20_001,
        metadrive_scenario_index=17,
        scenario_selection_seed=31,
        scenario_parameter_seed=37,
        case_id="level1_lead_brake",
        scenario_id="lead_brake",
        difficulty_level=1,
        requested_action=1,
        required_action=2,
        executed_action=2,
        unsafe_request=True,
        shield_intervened=True,
        shield_reasons=("minimum_ttc",),
        target_speed_mps=8.0,
        ego_speed_mps=10.0,
        ego_speed_limit_mps=13.0,
        ego_longitudinal_acceleration_mps2=-1.0,
        route_completion=0.2,
        route_progress_m=12.5,
        lane_offset_m=-0.1,
        collision_occurred=False,
        collision_kind=None,
        minimum_actual_ttc_s=2.4,
        minimum_actual_stopping_margin_m=3.5,
        pre_step_hard_rule_constraint=False,
        post_step_rule_violation_event=False,
        scenario_success=False,
        scenario_failure=False,
        arrived=False,
        off_road=False,
        terminated=step_index == 1,
        truncated=False,
        cumulative_unnecessary_stop_duration_s=0.0,
        reward_total=0.5,
        reward_components=components,
        claims=claims,
        review=CriticReview(
            conflict_score=0.65,
            unresolved_conflict=False,
            max_severity=0.8,
            supported_agent_ids=("hazard", "rule"),
            challenged_claim_ids=("nominal:aggregate",),
            reasons=("persisted disagreement",),
        ),
        expected_agent_ids=("nominal", "hazard", "rule"),
        failed_agent_ids=(),
        errors=(),
        policy_inference_latency_ms=1.0,
        agent_analysis_latency_ms=2.0,
        shield_latency_ms=0.5,
        total_decision_latency_ms=4.0,
        frame_path=frame_path,
    )


def finalize_manifest(bundle: Path) -> None:
    EvaluationWorkspace(destination=bundle, path=bundle).write_manifest()


def _eval_row(method_id: str, track: str, seed: str, *, disagreement: str) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in EVAL_METRICS_CSV_COLUMNS}
    row.update(
        {
            "result_label": SMOKE,
            "is_formal": "False",
            "record_schema_version": 1,
            "research_contract_version": 7,
            "track": track,
            "method_id": method_id,
            "policy_seed": seed,
            "case_id": "level1_lead_brake",
            "episode_index": 0,
            "test_seed": 20_001,
            "checkpoint_path": "" if method_id == "b0_rule" else f"runs/{method_id}/42.zip",
            "checkpoint_sha256": "" if method_id == "b0_rule" else "a" * 64,
            "policy_kind": "rule" if method_id == "b0_rule" else "ppo",
            "specialist_ids": "nominal;hazard;rule",
            "critic_enabled": method_id == "proposed",
            "shield_mode": "monitor" if track == "decision" else "enforce",
            "metadrive_scenario_index": 1,
            "scenario_selection_seed": 31,
            "scenario_parameter_seed": 37,
            "scenario_id": "lead_brake",
            "difficulty_level": 1,
            "collision": 0,
            "crossing_actor_collision": 0,
            "near_miss": 1,
            "minimum_actual_ttc_s": 2.4,
            "negative_stopping_margin": 0,
            "minimum_stopping_margin_m": 3.5,
            "hard_rule_violation": 0,
            "raw_unsafe_request_rate": 0.25,
            "shield_intervention_rate": 0.25,
            "off_road": 0,
            "scenario_success": 1,
            "final_route_completion": 0.8,
            "average_speed_mps": 8.5,
            "simulated_travel_time_s": 12.0,
            "unnecessary_braking_event_count": 1,
            "unnecessary_stop_duration_s": 0.5,
            "longitudinal_acceleration_rms_mps2": 0.8,
            "maximum_deceleration_mps2": 1.2,
            "longitudinal_jerk_rms_mps3": 0.4,
            "agent_disagreement_eligible_steps": 2,
            "agent_disagreement_count": 1,
            "agent_disagreement_rate": disagreement,
            "critic_challenge_eligible_steps": 2,
            "critic_challenge_count": 1,
            "critic_challenge_rate": 0.5,
            "critic_found_missed_danger_count": 1,
            "critic_found_missed_danger_rate": 0.5,
            "critic_false_challenge_count": 0,
            "critic_false_challenge_rate": 0.0,
            "agent_failure_fallback_count": 0,
            "decision_latency_p50_ms": 4.0,
            "decision_latency_p95_ms": 5.0,
            "decision_latency_p99_ms": 5.5,
            "episode_reward": 2.0,
        }
    )
    return row


def _comparison_rows() -> list[dict[str, object]]:
    metrics = (
        "collision",
        "scenario_success",
        "final_route_completion",
        "unnecessary_braking_event_count",
        "longitudinal_acceleration_rms_mps2",
        "agent_disagreement_rate",
        "decision_latency_p95_ms",
    )
    rows: list[dict[str, object]] = []
    for track, method in (
        ("decision", "b1_nominal"),
        ("system", "b0_rule"),
        ("ablation", "proposed"),
    ):
        for metric in metrics:
            rows.append(
                {
                    "result_label": SMOKE,
                    "is_formal": "False",
                    "track": track,
                    "case_id": "level1_lead_brake",
                    "method_id": method,
                    "metric": metric,
                    "physical_episode_count": 1,
                    "policy_replicate_count": 1,
                    "mean": (
                        "" if metric == "agent_disagreement_rate" and method == "b0_rule" else 0.5
                    ),
                    "policy_seed_stdev": "",
                }
            )
    return rows


@pytest.fixture
def bundle_factory(tmp_path: Path) -> Callable[[], Path]:
    bundle_index = 0

    def build() -> Path:
        nonlocal bundle_index
        bundle_index += 1
        name = "verified-bundle" if bundle_index == 1 else f"verified-bundle-{bundle_index}"
        bundle = tmp_path / name
        bundle.mkdir()
        bundle.joinpath("config_resolved.yaml").write_text(
            "method:\n  id: proposed\n", encoding="utf-8"
        )
        bundle.joinpath("evaluation_plan.yaml").write_text(
            "plan_kind: phase6_smoke\nevaluation_id: fixture\n", encoding="utf-8"
        )
        bundle.joinpath("model_selection.csv").write_text(
            "method_id,policy_seed,checkpoint_sha256\nproposed,42," + "a" * 64 + "\n",
            encoding="utf-8",
        )
        bundle.joinpath("selected_checkpoints.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "selected_checkpoints": [
                        {
                            "checkpoint_path": "runs/proposed/42.zip",
                            "checkpoint_sha256": "a" * 64,
                            "method_id": "proposed",
                            "policy_seed": 42,
                            "validation_plan_sha256": "b" * 64,
                        }
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        _write_csv(
            bundle / "metrics" / "train_metrics.csv",
            TRAIN_METRICS_CSV_COLUMNS,
            [
                {
                    "result_label": SMOKE,
                    "run_id": "proposed-42",
                    "method_id": "proposed",
                    "policy_seed": 42,
                    "timestep": timestep,
                    "metric": metric,
                    "value": value,
                }
                for timestep, metric, value in (
                    (10, "rollout/ep_rew_mean", 1.0),
                    (20, "rollout/ep_rew_mean", 2.0),
                    (0, "train/value_loss", ""),
                )
            ],
        )
        _write_csv(
            bundle / "metrics" / "eval_metrics.csv",
            EVAL_METRICS_CSV_COLUMNS,
            [
                _eval_row("b0_rule", "system", "", disagreement=""),
                _eval_row("b1_nominal", "decision", "42", disagreement="0.5"),
                _eval_row("proposed", "ablation", "42", disagreement="0.5"),
            ],
        )
        _write_csv(
            bundle / "metrics" / "comparison.csv",
            COMPARISON_CSV_COLUMNS,
            _comparison_rows(),
        )

        trace = (
            bundle
            / "episodes"
            / "proposed"
            / "system"
            / "42"
            / "level1_lead_brake"
            / "episode_20001_trace.jsonl"
        )
        frames = trace.parent / "episode_20001_frames"
        frames.mkdir(parents=True)
        frame_paths: list[str] = []
        for index, shade in enumerate((60, 140)):
            frame = frames / f"{index:06d}.png"
            iio.imwrite(frame, np.full((120, 220, 3), shade, dtype=np.uint8), extension=".png")
            frame_paths.append(frame.relative_to(bundle).as_posix())
        write_jsonl_strict(
            trace,
            [
                make_step(step_index=index, frame_path=frame_path).to_dict()
                for index, frame_path in enumerate(frame_paths)
            ],
        )
        trace.with_name("episode_20001_summary.json").write_text(
            '{"complete":true,"schema_version":1}\n', encoding="utf-8"
        )
        renders = bundle / "renders"
        renders.mkdir()
        iio.imwrite(
            renders / "proposed_42_level1_lead_brake_20001.gif",
            np.full((1, 16, 16, 3), 100, dtype=np.uint8),
            extension=".gif",
        )
        plots = bundle / "plots"
        plots.mkdir()
        for plot_name in (
            "learning_curve.png",
            "collision_rate.png",
            "success_route_completion.png",
            "unnecessary_braking.png",
            "comfort.png",
            "agent_disagreement.png",
        ):
            iio.imwrite(
                plots / plot_name,
                np.full((16, 16, 3), 80, dtype=np.uint8),
                extension=".png",
            )
        finalize_manifest(bundle)
        return bundle

    return build


@pytest.fixture
def step_bundle(bundle_factory: Callable[[], Path]) -> tuple[Path, Path, Path]:
    bundle = bundle_factory()
    trace = next(bundle.glob("episodes/**/*_trace.jsonl"))
    frames = trace.parent / "episode_20001_frames"
    return bundle, trace, frames


__all__ = ["SMOKE", "finalize_manifest", "make_step"]
