from __future__ import annotations

import csv
import io
import json
import os
import struct
import zipfile
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
    ValidationPhysicalIdentity,
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
from mad_driving.training.episode_seeds import (
    EpisodeSeedArtifactDescriptor,
    summarize_episode_seed_artifacts,
)
from mad_driving.training.metadata import (
    RunMetadata,
    checkpoint_curriculum_artifact_inventory,
    curriculum_state_artifact,
    write_run_metadata,
)

_TEST_MAX_SB3_DATA_BYTES = 16 * 1024 * 1024
_TEST_MAX_SB3_ARCHIVE_MEMBERS = 128


def _patch_data_member_headers(
    checkpoint: Path,
    *,
    set_encrypted: bool = False,
    compression: int | None = None,
    compressed_size: int | None = None,
    uncompressed_size: int | None = None,
) -> None:
    payload = bytearray(checkpoint.read_bytes())
    patched = 0
    for signature, name_length_offset, flag_offset, compression_offset, sizes_offset in (
        (b"PK\x03\x04", 26, 6, 8, 18),
        (b"PK\x01\x02", 28, 8, 10, 20),
    ):
        cursor = 0
        while (cursor := payload.find(signature, cursor)) >= 0:
            name_length = struct.unpack_from("<H", payload, cursor + name_length_offset)[0]
            name_start = cursor + (30 if signature == b"PK\x03\x04" else 46)
            name = bytes(payload[name_start : name_start + name_length])
            if name == b"data":
                if set_encrypted:
                    flags = struct.unpack_from("<H", payload, cursor + flag_offset)[0]
                    struct.pack_into("<H", payload, cursor + flag_offset, flags | 1)
                if compression is not None:
                    struct.pack_into("<H", payload, cursor + compression_offset, compression)
                if compressed_size is not None:
                    struct.pack_into("<I", payload, cursor + sizes_offset, compressed_size)
                if uncompressed_size is not None:
                    struct.pack_into("<I", payload, cursor + sizes_offset + 4, uncompressed_size)
                patched += 1
            cursor = name_start + name_length
    assert patched == 2
    checkpoint.write_bytes(payload)


def _corrupt_data_member_payload(checkpoint: Path) -> None:
    payload = bytearray(checkpoint.read_bytes())
    cursor = payload.find(b"PK\x03\x04")
    while cursor >= 0:
        name_length, extra_length = struct.unpack_from("<HH", payload, cursor + 26)
        name_start = cursor + 30
        name = bytes(payload[name_start : name_start + name_length])
        if name == b"data":
            compressed_size = struct.unpack_from("<I", payload, cursor + 18)[0]
            data_start = name_start + name_length + extra_length
            payload[data_start + compressed_size // 2] ^= 0x01
            checkpoint.write_bytes(payload)
            return
        cursor = payload.find(b"PK\x03\x04", name_start + name_length + extra_length)
    raise AssertionError("data local header not found")


def _write_test_sb3_checkpoint(
    checkpoint: Path,
    timestep: object,
    *,
    variant: str | None,
    duplicate_data: bool,
) -> None:
    if variant == "malformed_zip":
        checkpoint.write_bytes(b"not a ZIP")
        return
    data_name = "data/" if variant == "directory_data" else "data"
    if variant == "unsupported_compression":
        compression = zipfile.ZIP_BZIP2
    elif variant in ("high_compression_ratio", "oversized_actual_data"):
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED
    if isinstance(timestep, bytes):
        data = timestep
    else:
        content: dict[str, object] = {"num_timesteps": timestep}
        if variant == "high_compression_ratio":
            content["padding"] = "x" * (2 * 1024 * 1024)
        if variant == "oversized_actual_data":
            content["padding"] = "x" * _TEST_MAX_SB3_DATA_BYTES
        data = json.dumps(content).encode("utf-8")
    with zipfile.ZipFile(checkpoint, "w", compression=compression) as archive:
        if timestep is not None:
            archive.writestr(data_name, data)
            if duplicate_data:
                archive.writestr(data_name, data)
        archive.writestr("_stable_baselines3_version", "2.6.0")
        archive.writestr("system_info.txt", "python: test\n")
        if variant == "excessive_members":
            for index in range(_TEST_MAX_SB3_ARCHIVE_MEMBERS):
                archive.writestr(f"extra-{index:03d}", b"")
    if variant == "encrypted_data":
        _patch_data_member_headers(checkpoint, set_encrypted=True)
    elif variant == "malformed_compression":
        _patch_data_member_headers(checkpoint, compression=99)
    elif variant == "oversized_declared_data":
        _patch_data_member_headers(
            checkpoint,
            uncompressed_size=_TEST_MAX_SB3_DATA_BYTES + 1,
        )
    elif variant == "oversized_declared_compressed_data":
        _patch_data_member_headers(
            checkpoint,
            compressed_size=_TEST_MAX_SB3_DATA_BYTES + 1,
        )
    elif variant == "crc_error":
        _corrupt_data_member_payload(checkpoint)


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
    training_timestep: int = 100,
) -> CheckpointCandidate:
    return CheckpointCandidate(
        path=tmp_path / "checkpoints" / name,
        sha256=digest,
        method_id="proposed",
        policy_seed=policy_seed,
        checkpoint_kind="periodic",
        curriculum_level=3,
        training_timestep=training_timestep,
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
    checkpoint = candidate(
        tmp_path,
        digest,
        name,
        policy_seed=policy_seed,
        training_timestep=timestep,
    )
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
            episode_index=index,
            is_formal=True,
            shield_mode="enforce",
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
        validation_plan_rows=tuple(
            (record.episode.case_id, record.episode.episode_rng_seed) for record in episode_records
        ),
        episodes=tuple(episode_records),
    )


def with_physical_change(
    checkpoint_score: CheckpointScore,
    field_name: str,
) -> CheckpointScore:
    first = checkpoint_score.episodes[0]
    episode = replace(first.episode)
    replacements: dict[str, object] = {
        "scenario_id": "cut_in",
        "difficulty_level": 2,
        "metadrive_scenario_index": episode.metadrive_scenario_index + 1,
        "scenario_selection_seed": episode.scenario_selection_seed + 1,
        "scenario_parameter_seed": episode.scenario_parameter_seed + 1,
        "sampled_scenario_parameters": {
            "index": 0,
            "nested": {"actors": ["lead", "crossing"]},
        },
    }
    object.__setattr__(episode, field_name, replacements[field_name])
    changed = EpisodeMetricRecord(episode, first.metrics)
    return replace(
        checkpoint_score,
        episodes=(changed, *checkpoint_score.episodes[1:]),
    )


def test_checkpoint_score_binds_supplied_ordered_validation_plan_rows(
    tmp_path: Path,
) -> None:
    valid = score(
        tmp_path,
        digest="a" * 64,
        name="plan.zip",
        timestep=100,
        reward=1.0,
        collisions=0,
        successes=0,
        route_completion=0.1,
    )

    with pytest.raises(ValueError, match="validation plan rows"):
        replace(valid, validation_plan_rows=tuple(reversed(valid.validation_plan_rows)))


def test_checkpoint_score_rejects_duplicate_physical_identity(tmp_path: Path) -> None:
    valid = score(
        tmp_path,
        digest="a" * 64,
        name="duplicate.zip",
        timestep=100,
        reward=1.0,
        collisions=0,
        successes=0,
        route_completion=0.1,
    )

    with pytest.raises(ValueError, match="physical identity"):
        replace(
            valid,
            episodes=(*valid.episodes, valid.episodes[0]),
            validation_plan_rows=(*valid.validation_plan_rows, valid.validation_plan_rows[0]),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "scenario_id",
        "difficulty_level",
        "metadrive_scenario_index",
        "scenario_selection_seed",
        "scenario_parameter_seed",
        "sampled_scenario_parameters",
    ],
)
def test_selection_rejects_complete_physical_identity_drift_before_ranking(
    tmp_path: Path,
    field_name: str,
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
        route_completion=0.2,
    )
    second = with_physical_change(second, field_name)

    with pytest.raises(ValueError, match="physical identities"):
        select_checkpoint((first, second))


def test_physical_identity_canonicalizes_nested_sampled_parameters(
    tmp_path: Path,
) -> None:
    checkpoint_score = score(
        tmp_path,
        digest="a" * 64,
        name="canonical.zip",
        timestep=100,
        reward=1.0,
        collisions=0,
        successes=0,
        route_completion=0.1,
    )

    identity = checkpoint_score.physical_identities[0]

    assert type(identity) is ValidationPhysicalIdentity
    assert identity.canonical_sampled_scenario_parameters == '{"index":0}'


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
    checkpoint_data_payloads: dict[str, object] | None = None,
    malformed_checkpoints: frozenset[str] = frozenset(),
    duplicate_data_members: frozenset[str] = frozenset(),
    checkpoint_variants: dict[str, str] | None = None,
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

    checkpoints: list[tuple[str, CurriculumState, int]] = [("best_model.zip", state, 5_000)]
    if include_supported_checkpoints:
        checkpoints.extend(
            (
                (
                    "best_model_level_1.zip",
                    CurriculumState(level=2, consecutive_passes=0, evaluations=4),
                    2_000,
                ),
                ("final_model.zip", state, 5_000),
                (
                    "ppo_checkpoint_2500_steps.zip",
                    CurriculumState(level=1, consecutive_passes=0, evaluations=2),
                    2_500,
                ),
            )
        )
    if include_unknown_listed_checkpoint:
        checkpoints.append(("researcher_favorite.zip", state, 5_000))
    for name, checkpoint_state, default_timestep in checkpoints:
        checkpoint = checkpoints_dir / name
        configured = (checkpoint_data_payloads or {}).get(name, default_timestep)
        variant = (checkpoint_variants or {}).get(name)
        if name in malformed_checkpoints:
            variant = "malformed_zip"
        _write_test_sb3_checkpoint(
            checkpoint,
            configured,
            variant=variant,
            duplicate_data=name in duplicate_data_members,
        )
        write_checkpoint_curriculum_state(checkpoint_state, checkpoint)

    seed_artifact = seeds_dir / "train-worker-000.jsonl"
    seed_artifact.touch()
    seed_metadata = seed_artifact.stat()
    seed_records = (
        {
            "file_identity": {
                "device": seed_metadata.st_dev,
                "inode": seed_metadata.st_ino,
            },
            "record_type": "episode_seed_artifact",
            "role": "train",
            "schema_version": 4,
            "worker_index": 0,
        },
        {
            "role": "train",
            "worker_index": 0,
            "environment_seed": 101,
            "scenario_selection_seed": 102,
            "scenario_parameter_seed": 103,
            "scenario_id": "lead_brake",
            "difficulty_level": 1,
            "scenario_parameters": {"initial_gap_m": 30.0},
        },
    )
    seed_artifact.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in seed_records
        ),
        encoding="utf-8",
    )
    episode_seed_artifacts = summarize_episode_seed_artifacts(
        run_dir,
        expected_descriptors=(
            EpisodeSeedArtifactDescriptor(
                role="train",
                worker_index=0,
                relative_path="episode_seeds/train-worker-000.jsonl",
                device=seed_metadata.st_dev,
                inode=seed_metadata.st_ino,
            ),
        ),
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
    assert tuple(candidate.training_timestep for candidate in candidates) == (2_000, 5_000, 2_500)
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
        ("missing_seed_artifact", "seed artifact|inventory|environment"),
        ("replaced_seed_artifact", "seed artifact|identity"),
        ("extra_seed_artifact", "seed artifact|inventory|environment"),
        ("malformed_seed_artifact", "seed artifact|malformed"),
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
    elif corruption == "missing_seed_artifact":
        (run_dir / "episode_seeds" / "train-worker-000.jsonl").unlink()
    elif corruption == "replaced_seed_artifact":
        seed_artifact = run_dir / "episode_seeds" / "train-worker-000.jsonl"
        seed_artifact.unlink()
        seed_artifact.write_text("replacement\n", encoding="utf-8")
    elif corruption == "extra_seed_artifact":
        (run_dir / "episode_seeds" / "extra.jsonl").write_text("extra\n", encoding="utf-8")
    elif corruption == "malformed_seed_artifact":
        (run_dir / "episode_seeds" / "train-worker-000.jsonl").write_text(
            "{malformed\n", encoding="utf-8"
        )
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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "data"),
        (b"{malformed", "data"),
        (True, "num_timesteps"),
        (-1, "num_timesteps"),
        (1.5, "num_timesteps"),
        ("5000", "num_timesteps"),
    ],
)
def test_discovery_rejects_missing_malformed_or_invalid_sb3_timestep_data(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    run_dir = completed_training_run(
        tmp_path,
        checkpoint_data_payloads={"final_model.zip": payload},
    )

    with pytest.raises(ValueError, match=message):
        discover_checkpoint_candidates(run_dir)


def test_discovery_rejects_malformed_zip_and_duplicate_data_member(tmp_path: Path) -> None:
    malformed = completed_training_run(
        tmp_path / "malformed",
        malformed_checkpoints=frozenset({"final_model.zip"}),
    )
    duplicate = completed_training_run(
        tmp_path / "duplicate",
        duplicate_data_members=frozenset({"final_model.zip"}),
    )

    with pytest.raises(ValueError, match="ZIP|checkpoint"):
        discover_checkpoint_candidates(malformed)
    with pytest.raises(ValueError, match="data"):
        discover_checkpoint_candidates(duplicate)


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("oversized_declared_data", "size|large|limit"),
        ("oversized_declared_compressed_data", "compressed|size|large|limit"),
        ("oversized_actual_data", "size|large|limit"),
        ("high_compression_ratio", "ratio|compression"),
        ("excessive_members", "member|entries"),
        ("encrypted_data", "encrypted"),
        ("directory_data", "directory|data"),
        ("unsupported_compression", "compression"),
        ("malformed_compression", "compression|malformed"),
        ("crc_error", "CRC|decompression|malformed"),
    ],
)
def test_discovery_rejects_resource_unsafe_sb3_archives(
    tmp_path: Path,
    variant: str,
    message: str,
) -> None:
    run_dir = completed_training_run(
        tmp_path,
        checkpoint_variants={"best_model.zip": variant},
    )

    with pytest.raises(ValueError, match=message):
        discover_checkpoint_candidates(run_dir)


def test_discovery_caps_actual_streamed_sb3_data_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = completed_training_run(tmp_path)
    original_open = zipfile.ZipFile.open

    def oversized_open(
        archive: zipfile.ZipFile,
        name: str | zipfile.ZipInfo,
        *args: object,
        **kwargs: object,
    ) -> object:
        member_name = name.filename if isinstance(name, zipfile.ZipInfo) else name
        if Path(archive.filename).name == "best_model.zip" and member_name == "data":
            return io.BytesIO(b"x" * (_TEST_MAX_SB3_DATA_BYTES + 1))
        return original_open(archive, name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(zipfile.ZipFile, "open", oversized_open)

    with pytest.raises(ValueError, match="streamed|actual|limit"):
        discover_checkpoint_candidates(run_dir)


def test_discovery_rejects_checkpoint_path_identity_change_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = completed_training_run(tmp_path)
    target = (run_dir / "checkpoints" / "best_model.zip").absolute()
    original_open = Path.open
    original_lstat = Path.lstat
    parser_opened = False

    class ChangedIdentity:
        def __init__(self, source: os.stat_result) -> None:
            self._source = source

        @property
        def st_ino(self) -> int:
            return int(self._source.st_ino) + 1

        def __getattr__(self, name: str) -> object:
            return getattr(self._source, name)

    def racing_open(path: Path, *args: object, **kwargs: object) -> object:
        nonlocal parser_opened
        source = original_open(path, *args, **kwargs)  # type: ignore[arg-type]
        mode = args[0] if args else kwargs.get("mode", "r")
        if path.absolute() == target and mode == "rb" and kwargs.get("buffering") == 0:
            parser_opened = True
        return source

    def racing_lstat(path: Path) -> object:
        metadata = original_lstat(path)
        if parser_opened and path.absolute() == target:
            return ChangedIdentity(metadata)
        return metadata

    monkeypatch.setattr(Path, "open", racing_open)
    monkeypatch.setattr(Path, "lstat", racing_lstat)

    with pytest.raises(ValueError, match="identity|replaced|changed"):
        discover_checkpoint_candidates(run_dir)


def test_discovery_hashes_the_same_opened_checkpoint_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = completed_training_run(tmp_path)
    target = (run_dir / "checkpoints" / "best_model.zip").absolute()
    original_open = Path.open
    changed = False

    def corrupting_open(path: Path, *args: object, **kwargs: object) -> object:
        nonlocal changed
        source = original_open(path, *args, **kwargs)  # type: ignore[arg-type]
        mode = args[0] if args else kwargs.get("mode", "r")
        if (
            not changed
            and path.absolute() == target
            and mode == "rb"
            and kwargs.get("buffering") == 0
        ):
            source.close()
            with original_open(path, "r+b") as mutable:
                first_byte = mutable.read(1)
                mutable.seek(0)
                mutable.write(bytes((first_byte[0] ^ 1,)))
                mutable.flush()
            source = original_open(path, *args, **kwargs)  # type: ignore[arg-type]
            changed = True
        return source

    monkeypatch.setattr(Path, "open", corrupting_open)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        discover_checkpoint_candidates(run_dir)


def test_discovery_rejects_periodic_filename_timestep_mismatch(tmp_path: Path) -> None:
    run_dir = completed_training_run(
        tmp_path,
        checkpoint_data_payloads={"ppo_checkpoint_2500_steps.zip": 2_499},
    )

    with pytest.raises(ValueError, match="filename|timestep"):
        discover_checkpoint_candidates(run_dir)


def test_checkpoint_score_rejects_caller_timestep_mismatch(tmp_path: Path) -> None:
    valid = score(
        tmp_path,
        digest="a" * 64,
        name="mismatch.zip",
        timestep=100,
        reward=1.0,
        collisions=0,
        successes=0,
        route_completion=0.1,
    )

    with pytest.raises(ValueError, match="authenticated candidate"):
        replace(valid, training_timestep=101)


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


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("episode_reward", nan),
        ("decision_latency_p99_ms", -1.0),
        ("agent_failure_fallback_count", -1),
        ("collision", 1),
    ],
)
def test_checkpoint_score_rejects_malformed_metrics(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
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
    object.__setattr__(first.metrics, field_name, value)

    with pytest.raises((TypeError, ValueError)):
        replace(valid)


@pytest.mark.parametrize("mismatch", ["plan", "matrix", "physical"])
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
    elif mismatch == "physical":
        second = with_physical_change(second, "sampled_scenario_parameters")
    output_dir = tmp_path / "cross-group"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="validation plan|scenario/seed matrix"):
        write_selection_artifacts(output_dir, (first, second))
    assert not (output_dir / "model_selection.csv").exists()
    assert not (output_dir / "selected_checkpoints.json").exists()
