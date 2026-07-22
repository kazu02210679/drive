from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from math import nan
from pathlib import Path

import pytest
import yaml

from mad_driving.evaluation.metrics import EpisodeMetricRecord, EpisodeMetrics
from mad_driving.evaluation.models import (
    EVALUATION_CASES,
    EvaluationEpisodeKey,
    EvaluationEpisodeRecord,
)
from mad_driving.evaluation.selection import (
    CheckpointCandidate,
    CheckpointScore,
    discover_checkpoint_candidates,
    select_checkpoint,
    write_selection_artifacts,
)
from mad_driving.methods import MethodProfileSnapshot
from mad_driving.training.curriculum import (
    CurriculumState,
    write_checkpoint_curriculum_state,
    write_curriculum_state,
)
from mad_driving.training.metadata import (
    RunMetadata,
    checkpoint_curriculum_artifact_inventory,
    curriculum_state_artifact,
    write_run_metadata,
)


def metrics(
    *,
    reward: float,
    collision: bool,
    success: bool,
    route_completion: float,
) -> EpisodeMetrics:
    return EpisodeMetrics(
        collision=collision,
        crossing_actor_collision=False,
        near_miss=False,
        minimum_actual_ttc_s=None,
        negative_stopping_margin=False,
        minimum_stopping_margin_m=None,
        hard_rule_violation=False,
        raw_unsafe_request_rate=0.0,
        shield_intervention_rate=0.0,
        off_road=False,
        scenario_success=success,
        final_route_completion=route_completion,
        average_speed_mps=5.0,
        simulated_travel_time_s=10.0,
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
        decision_latency_p50_ms=1.0,
        decision_latency_p95_ms=1.0,
        decision_latency_p99_ms=1.0,
        episode_reward=reward,
    )


def candidate(
    tmp_path: Path,
    digest: str,
    name: str,
    *,
    policy_seed: int = 42,
) -> CheckpointCandidate:
    return CheckpointCandidate(
        path=tmp_path / "checkpoints" / name,
        sha256=digest,
        method_id="proposed",
        policy_seed=policy_seed,
        checkpoint_kind="periodic",
        curriculum_level=3,
    )


def score(
    tmp_path: Path,
    *,
    digest: str,
    name: str,
    timestep: int,
    reward: float,
    collisions: int,
    successes: int,
    route_completion: float,
    validation_seed: int = 10_001,
    role: str = "validation",
    policy_seed: int = 42,
) -> CheckpointScore:
    checkpoint = candidate(tmp_path, digest, name, policy_seed=policy_seed)
    profile = MethodProfileSnapshot.from_method_id("proposed")
    episode_records: list[EpisodeMetricRecord] = []
    for index, case in enumerate(EVALUATION_CASES):
        episode_seed = validation_seed if role == "validation" else 20_001
        episode_key = EvaluationEpisodeKey(
            method_id="proposed",
            track="system",
            role=role,  # type: ignore[arg-type]
            policy_seed=policy_seed,
            case_id=case.case_id,
            episode_rng_seed=episode_seed,
        )
        episode = EvaluationEpisodeRecord(
            record_schema_version=1,
            research_contract_version=7,
            episode_key=episode_key,
            method_profile=profile,
            checkpoint_path=str(checkpoint.path),
            checkpoint_sha256=checkpoint.sha256,
            episode_rng_seed=episode_seed,
            metadrive_scenario_index=100 + index,
            scenario_selection_seed=200 + index,
            scenario_parameter_seed=300 + index,
            case_id=case.case_id,
            scenario_id=case.scenario_id,
            difficulty_level=case.difficulty_level,
            sampled_scenario_parameters={"index": index},
            step_count=1,
            final_step_index=0,
            simulated_duration_s=0.1,
            cumulative_reward=reward,
            collision_occurred=index < collisions,
            collision_kind="vehicle" if index < collisions else None,
            scenario_success=index < successes,
            scenario_failure=index >= successes,
            arrived=index < successes,
            off_road=False,
            terminated=True,
            truncated=False,
            complete=True,
        )
        episode_records.append(
            EpisodeMetricRecord(
                episode,
                metrics(
                    reward=reward,
                    collision=index < collisions,
                    success=index < successes,
                    route_completion=route_completion,
                ),
            )
        )
    return CheckpointScore(
        candidate=checkpoint,
        training_timestep=timestep,
        validation_plan_sha256="f" * 64,
        episodes=tuple(episode_records),
    )


def test_select_checkpoint_uses_all_six_lexicographic_keys(tmp_path: Path) -> None:
    scores = (
        score(
            tmp_path,
            digest="1" * 64,
            name="low-reward.zip",
            timestep=100,
            reward=9.0,
            collisions=0,
            successes=5,
            route_completion=1.0,
        ),
        score(
            tmp_path,
            digest="2" * 64,
            name="high-collision.zip",
            timestep=100,
            reward=10.0,
            collisions=2,
            successes=5,
            route_completion=1.0,
        ),
        score(
            tmp_path,
            digest="3" * 64,
            name="low-success.zip",
            timestep=100,
            reward=10.0,
            collisions=1,
            successes=2,
            route_completion=1.0,
        ),
        score(
            tmp_path,
            digest="4" * 64,
            name="low-route.zip",
            timestep=100,
            reward=10.0,
            collisions=1,
            successes=3,
            route_completion=0.5,
        ),
        score(
            tmp_path,
            digest="e" * 64,
            name="late.zip",
            timestep=300,
            reward=10.0,
            collisions=1,
            successes=3,
            route_completion=0.7,
        ),
        score(
            tmp_path,
            digest="d" * 64,
            name="early-large-hash.zip",
            timestep=200,
            reward=10.0,
            collisions=1,
            successes=3,
            route_completion=0.7,
        ),
        score(
            tmp_path,
            digest="a" * 64,
            name="selected.zip",
            timestep=200,
            reward=10.0,
            collisions=1,
            successes=3,
            route_completion=0.7,
        ),
    )

    selected = select_checkpoint(tuple(reversed(scores)))

    assert selected.candidate.path.name == "selected.zip"
    assert selected.mean_episode_reward == 10.0
    assert selected.collision_rate == 0.2
    assert selected.success_rate == 0.6
    assert selected.mean_route_completion == 0.7


def test_select_checkpoint_rejects_different_ordered_validation_matrices(
    tmp_path: Path,
) -> None:
    first = score(
        tmp_path,
        digest="a" * 64,
        name="first.zip",
        timestep=100,
        reward=1.0,
        collisions=0,
        successes=0,
        route_completion=0.1,
    )
    second = score(
        tmp_path,
        digest="b" * 64,
        name="second.zip",
        timestep=200,
        reward=2.0,
        collisions=0,
        successes=0,
        route_completion=0.1,
        validation_seed=10_002,
    )

    with pytest.raises(ValueError, match="scenario/seed matrix"):
        select_checkpoint((first, second))


def test_checkpoint_score_rejects_any_test_split_episode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="validation"):
        score(
            tmp_path,
            digest="a" * 64,
            name="test-data.zip",
            timestep=100,
            reward=1.0,
            collisions=0,
            successes=0,
            route_completion=0.1,
            role="test",
        )


def completed_training_run(
    tmp_path: Path,
    *,
    complete: bool = True,
    include_supported_checkpoints: bool = True,
    include_unknown_listed_checkpoint: bool = False,
) -> Path:
    run_dir = tmp_path / "run"
    checkpoints_dir = run_dir / "checkpoints"
    seeds_dir = run_dir / "episode_seeds"
    checkpoints_dir.mkdir(parents=True)
    seeds_dir.mkdir()
    resolved_config = {"method": {"id": "proposed"}, "training": {"seed": 42}}
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=True),
        encoding="utf-8",
    )
    state = CurriculumState(level=3, consecutive_passes=0, evaluations=8)
    state_path = run_dir / "curriculum_state.yaml"
    write_curriculum_state(state, state_path)

    checkpoints: list[tuple[str, CurriculumState]] = [("best_model.zip", state)]
    if include_supported_checkpoints:
        checkpoints.extend(
            (
                (
                    "best_model_level_1.zip",
                    CurriculumState(level=2, consecutive_passes=0, evaluations=4),
                ),
                ("final_model.zip", state),
                (
                    "ppo_checkpoint_2500_steps.zip",
                    CurriculumState(level=1, consecutive_passes=0, evaluations=2),
                ),
            )
        )
    if include_unknown_listed_checkpoint:
        checkpoints.append(("researcher_favorite.zip", state))
    for name, checkpoint_state in checkpoints:
        checkpoint = checkpoints_dir / name
        checkpoint.write_bytes(f"authenticated:{name}".encode())
        write_checkpoint_curriculum_state(checkpoint_state, checkpoint)

    seed_artifact = seeds_dir / "train-worker-000.jsonl"
    seed_artifact.write_text('{"seed":1}\n', encoding="utf-8")
    metadata = seed_artifact.stat()
    episode_seed_artifacts = (
        {
            "file_identity": {"device": metadata.st_dev, "inode": metadata.st_ino},
            "path": "episode_seeds/train-worker-000.jsonl",
            "record_count": 1,
            "role": "train",
            "schema_version": 4,
            "sha256": hashlib.sha256(seed_artifact.read_bytes()).hexdigest(),
            "worker_index": 0,
        },
    )
    write_run_metadata(
        RunMetadata(
            resolved_config=resolved_config,
            method_profile=MethodProfileSnapshot.from_method_id("proposed"),
            curriculum_state=curriculum_state_artifact(state_path, state),
            checkpoint_curriculum_artifacts=(
                checkpoint_curriculum_artifact_inventory(checkpoints_dir) if complete else ()
            ),
            episode_seed_artifacts=episode_seed_artifacts if complete else (),
        ),
        run_dir / "run_metadata.json",
    )
    return run_dir


def test_discovery_accepts_only_supported_candidates_from_verified_inventory(
    tmp_path: Path,
) -> None:
    run_dir = completed_training_run(tmp_path)

    candidates = discover_checkpoint_candidates(run_dir)

    assert tuple(candidate.path.name for candidate in candidates) == (
        "best_model_level_1.zip",
        "final_model.zip",
        "ppo_checkpoint_2500_steps.zip",
    )
    assert tuple(candidate.checkpoint_kind for candidate in candidates) == (
        "level_best",
        "final",
        "periodic",
    )
    assert tuple(candidate.curriculum_level for candidate in candidates) == (2, 3, 1)
    assert {candidate.method_id for candidate in candidates} == {"proposed"}
    assert {candidate.policy_seed for candidate in candidates} == {42}
    assert all(candidate.path.is_absolute() for candidate in candidates)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("unlisted_zip", "inventory|sidecar"),
        ("missing_sidecar", "sidecar"),
        ("invalid_sidecar", "curriculum|sidecar|malformed"),
        ("checkpoint_hash", "SHA-256|hash|inventory|different checkpoint"),
        ("method", "method"),
        ("seed", "seed"),
        ("schema", "research_contract_version|contract"),
    ],
)
def test_discovery_rejects_corrupt_or_mismatched_training_runs(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    run_dir = completed_training_run(tmp_path)
    checkpoint = run_dir / "checkpoints" / "final_model.zip"
    sidecar = checkpoint.with_name(f"{checkpoint.name}.curriculum.yaml")
    if corruption == "unlisted_zip":
        (run_dir / "checkpoints" / "unlisted.zip").write_bytes(b"not inventoried")
    elif corruption == "missing_sidecar":
        sidecar.unlink()
    elif corruption == "invalid_sidecar":
        sidecar.write_text("schema_version: [\n", encoding="utf-8")
    elif corruption == "checkpoint_hash":
        checkpoint.write_bytes(b"replacement")
    elif corruption in ("method", "seed"):
        config_path = run_dir / "config_resolved.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if corruption == "method":
            config["method"]["id"] = "b1_nominal"
        else:
            config["training"]["seed"] = 43
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    else:
        metadata_path = run_dir / "run_metadata.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["research_contract_version"] = 6
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        discover_checkpoint_candidates(run_dir)


def test_discovery_rejects_incomplete_or_unknown_inventory_entries(tmp_path: Path) -> None:
    incomplete = completed_training_run(tmp_path / "incomplete", complete=False)
    unknown = completed_training_run(
        tmp_path / "unknown",
        include_unknown_listed_checkpoint=True,
    )

    with pytest.raises(ValueError, match="complete"):
        discover_checkpoint_candidates(incomplete)
    with pytest.raises(ValueError, match="unsupported|candidate"):
        discover_checkpoint_candidates(unknown)


def test_selection_artifacts_record_all_scores_and_full_selected_identity(
    tmp_path: Path,
) -> None:
    lower = score(
        tmp_path,
        digest="b" * 64,
        name="lower.zip",
        timestep=100,
        reward=1.0,
        collisions=0,
        successes=1,
        route_completion=0.2,
    )
    selected = score(
        tmp_path,
        digest="a" * 64,
        name="selected.zip",
        timestep=200,
        reward=2.0,
        collisions=1,
        successes=3,
        route_completion=0.7,
    )
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()

    csv_path, json_path = write_selection_artifacts(output_dir, (selected, lower))

    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
        assert source.newlines == "\r\n" or source.newlines == "\n"
    assert tuple(rows[0]) == (
        "method_id",
        "policy_seed",
        "checkpoint_path",
        "checkpoint_sha256",
        "training_timestep",
        "validation_plan_sha256",
        "mean_episode_reward",
        "collision_rate",
        "success_rate",
        "mean_route_completion",
        "selected",
    )
    assert [row["checkpoint_path"] for row in rows] == [
        str(lower.candidate.path),
        str(selected.candidate.path),
    ]
    assert [row["selected"] for row in rows] == ["false", "true"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "selected_checkpoints": [
            {
                "checkpoint_path": str(selected.candidate.path),
                "checkpoint_sha256": "a" * 64,
                "method_id": "proposed",
                "policy_seed": 42,
                "validation_plan_sha256": "f" * 64,
            }
        ],
    }
    assert csv_path.read_bytes().endswith(b"\n")
    assert json_path.read_bytes().endswith(b"\n")


def test_checkpoint_score_rejects_non_finite_ranking_metrics(tmp_path: Path) -> None:
    valid = score(
        tmp_path,
        digest="a" * 64,
        name="nonfinite.zip",
        timestep=100,
        reward=1.0,
        collisions=0,
        successes=0,
        route_completion=0.1,
    )
    first = valid.episodes[0]
    nonfinite = EpisodeMetricRecord(
        first.episode,
        replace(first.metrics, episode_reward=nan),
    )

    with pytest.raises(ValueError, match="finite"):
        CheckpointScore(
            candidate=valid.candidate,
            training_timestep=valid.training_timestep,
            validation_plan_sha256=valid.validation_plan_sha256,
            episodes=(nonfinite, *valid.episodes[1:]),
        )


@pytest.mark.parametrize("mismatch", ["plan", "matrix"])
def test_selection_artifacts_reject_cross_group_validation_mismatches(
    tmp_path: Path,
    mismatch: str,
) -> None:
    first = score(
        tmp_path,
        digest="a" * 64,
        name="seed-42.zip",
        timestep=100,
        reward=1.0,
        collisions=0,
        successes=0,
        route_completion=0.1,
    )
    second = score(
        tmp_path,
        digest="b" * 64,
        name="seed-43.zip",
        timestep=100,
        reward=1.0,
        collisions=0,
        successes=0,
        route_completion=0.1,
        policy_seed=43,
        validation_seed=10_002 if mismatch == "matrix" else 10_001,
    )
    if mismatch == "plan":
        second = replace(second, validation_plan_sha256="e" * 64)
    output_dir = tmp_path / "cross-group"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="validation plan|scenario/seed matrix"):
        write_selection_artifacts(output_dir, (first, second))
