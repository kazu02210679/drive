import json
import math
import pickle
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from numpy.typing import NDArray

import mad_driving.training.episode_seeds as episode_seed_module
from mad_driving.training.episode_seeds import (
    EpisodeSeedArtifactDescriptor,
    EpisodeSeedRecordingWrapper,
    summarize_episode_seed_artifacts,
)


class ResetInfoEnv(gym.Env[NDArray[np.float32], int]):
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
    action_space = spaces.Discrete(4)

    def __init__(self, reset_infos: list[dict[str, Any]]) -> None:
        self.reset_infos = reset_infos
        self.reset_calls = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        info = self.reset_infos[self.reset_calls]
        self.reset_calls += 1
        return np.zeros(24, dtype=np.float32), info

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        del action
        return np.zeros(24, dtype=np.float32), 0.0, False, False, {}


def seed_info(episode: int, road: int, parameters: int) -> dict[str, int]:
    return {
        "episode_rng_seed": episode,
        "metadrive_scenario_index": road,
        "scenario_parameter_seed": parameters,
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [value for value in values if "episode_rng_seed" in value]


def replace_artifact_if_permitted(path: Path, replacement: bytes) -> bool:
    displaced = path.parent.parent / f".{path.name}.displaced"
    try:
        path.replace(displaced)
        path.write_bytes(replacement)
    except OSError:
        return False
    return True


def descriptor_for_artifact(
    path: Path,
    *,
    role: str,
    worker_index: int,
) -> EpisodeSeedArtifactDescriptor:
    stat_result = path.stat()
    return EpisodeSeedArtifactDescriptor(
        role=role,  # type: ignore[arg-type]
        worker_index=worker_index,
        relative_path=f"episode_seeds/{path.name}",
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
    )


def test_replacement_after_exclusive_create_cannot_inject_history(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(2, 3, 4)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )
    artifact = workspace / "episode_seeds" / "train-worker-000.jsonl"
    injected = (
        json.dumps(
            {
                "episode_rng_seed": 1,
                "metadrive_scenario_index": 2,
                "role": "train",
                "scenario_parameter_seed": 3,
                "worker_index": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()

    replaced = replace_artifact_if_permitted(artifact, injected)
    if not replaced:
        wrapped.reset()
        wrapped.close()
        return

    with pytest.raises(RuntimeError, match="identity"):
        wrapped.reset()
    wrapped.close()


def test_replacement_between_appends_cannot_inject_or_lose_history(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3), seed_info(4, 5, 6)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )
    artifact = workspace / "episode_seeds" / "train-worker-000.jsonl"
    wrapped.reset()
    first_history = artifact.read_bytes()

    replaced = replace_artifact_if_permitted(artifact, first_history)
    if not replaced:
        wrapped.reset()
        wrapped.close()
        return

    with pytest.raises(RuntimeError, match="identity"):
        wrapped.reset()
    wrapped.close()


def test_replacement_between_parse_and_hash_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )
    wrapped.reset()
    trusted_descriptor = wrapped.episode_seed_artifact_descriptor
    wrapped.close()
    artifact = workspace / "episode_seeds" / "train-worker-000.jsonl"
    original = artifact.read_bytes()
    real_loads = episode_seed_module.json.loads
    replacement_attempted = False
    replacement_succeeded = False

    def replace_after_record_parse(*args: Any, **kwargs: Any) -> Any:
        nonlocal replacement_attempted, replacement_succeeded
        parsed = real_loads(*args, **kwargs)
        if not replacement_attempted and isinstance(parsed, dict) and "episode_rng_seed" in parsed:
            replacement_attempted = True
            replacement_succeeded = replace_artifact_if_permitted(artifact, original)
        return parsed

    monkeypatch.setattr(episode_seed_module.json, "loads", replace_after_record_parse)

    try:
        summarize_episode_seed_artifacts(
            workspace,
            expected_descriptors=(trusted_descriptor,),
        )
    except ValueError as error:
        assert replacement_succeeded
        assert "identity" in str(error) or "changed" in str(error)
    else:
        assert replacement_attempted
        assert not replacement_succeeded


def test_parent_held_identity_rejects_post_close_self_attested_replacement(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )
    trusted_descriptor = wrapped.episode_seed_artifact_descriptor
    wrapped.reset()
    wrapped.close()

    artifact = workspace / "episode_seeds" / "train-worker-000.jsonl"
    artifact.replace(workspace / "displaced-original.jsonl")
    artifact.write_bytes(b"placeholder\n")
    replacement_stat = artifact.stat()
    replacement_header = {
        "file_identity": {
            "device": replacement_stat.st_dev,
            "inode": replacement_stat.st_ino,
        },
        "record_type": "episode_seed_artifact",
        "role": "train",
        "schema_version": 2,
        "worker_index": 0,
    }
    replacement_record = {
        "episode_rng_seed": 999,
        "metadrive_scenario_index": 1000,
        "role": "train",
        "scenario_parameter_seed": 1001,
        "worker_index": 0,
    }
    artifact.write_text(
        "\n".join(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in (replacement_header, replacement_record)
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity"):
        summarize_episode_seed_artifacts(
            workspace,
            expected_descriptors=(trusted_descriptor,),
        )


def test_parent_held_inventory_rejects_a_missing_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )
    trusted_descriptor = wrapped.episode_seed_artifact_descriptor
    wrapped.close()
    artifact = workspace / trusted_descriptor.relative_path
    artifact.replace(workspace / "missing-from-inventory.jsonl")

    with pytest.raises(ValueError, match="inventory"):
        summarize_episode_seed_artifacts(
            workspace,
            expected_descriptors=(trusted_descriptor,),
        )


def test_live_writer_descriptor_is_immutable_json_and_pickle_safe(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )

    descriptor = wrapped.episode_seed_artifact_descriptor

    assert json.loads(json.dumps(descriptor)) == [
        "train",
        0,
        "episode_seeds/train-worker-000.jsonl",
        descriptor.device,
        descriptor.inode,
    ]
    assert pickle.loads(pickle.dumps(descriptor)) == descriptor
    with pytest.raises(AttributeError):
        descriptor.role = "validation"  # type: ignore[misc]
    wrapped.close()


def test_inventory_rejects_duplicate_parent_descriptors(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )
    trusted_descriptor = wrapped.episode_seed_artifact_descriptor
    wrapped.close()

    with pytest.raises(ValueError, match="duplicate"):
        summarize_episode_seed_artifacts(
            workspace,
            expected_descriptors=(trusted_descriptor, trusted_descriptor),
        )


def test_inventory_rejects_a_mismatched_parent_worker(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )
    trusted_descriptor = wrapped.episode_seed_artifact_descriptor
    wrapped.close()

    with pytest.raises(ValueError, match="descriptor path"):
        summarize_episode_seed_artifacts(
            workspace,
            expected_descriptors=(trusted_descriptor._replace(worker_index=1),),
        )


def test_inventory_rejects_a_non_descriptor_parent_value(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="descriptor is malformed"):
        summarize_episode_seed_artifacts(
            workspace,
            expected_descriptors=(object(),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "identity_changes",
    [
        {"device": True},
        {"inode": 0},
    ],
)
def test_inventory_rejects_a_malformed_parent_file_identity(
    tmp_path: Path,
    identity_changes: dict[str, object],
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )
    trusted_descriptor = wrapped.episode_seed_artifact_descriptor
    wrapped.close()

    with pytest.raises(ValueError, match="descriptor identity"):
        summarize_episode_seed_artifacts(
            workspace,
            expected_descriptors=(trusted_descriptor._replace(**identity_changes),),
        )


def test_records_each_actual_reset_info_in_order_and_summarizes_it(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    environment = ResetInfoEnv([seed_info(41, 101, 201), seed_info(42, 102, 202)])
    wrapped = EpisodeSeedRecordingWrapper(
        environment,
        workspace=workspace,
        role="train",
        worker_index=0,
    )

    wrapped.reset(seed=999)
    wrapped.reset()
    trusted_descriptor = wrapped.episode_seed_artifact_descriptor
    wrapped.close()

    artifact = workspace / "episode_seeds" / "train-worker-000.jsonl"
    assert read_jsonl(artifact) == [
        {
            "episode_rng_seed": 41,
            "metadrive_scenario_index": 101,
            "role": "train",
            "scenario_parameter_seed": 201,
            "worker_index": 0,
        },
        {
            "episode_rng_seed": 42,
            "metadrive_scenario_index": 102,
            "role": "train",
            "scenario_parameter_seed": 202,
            "worker_index": 0,
        },
    ]
    summaries = summarize_episode_seed_artifacts(
        workspace,
        expected_descriptors=(trusted_descriptor,),
    )
    assert summaries == (
        {
            "file_identity": summaries[0]["file_identity"],
            "path": "episode_seeds/train-worker-000.jsonl",
            "record_count": 2,
            "role": "train",
            "schema_version": 2,
            "sha256": summaries[0]["sha256"],
            "worker_index": 0,
        },
    )
    assert len(summaries[0]["sha256"]) == 64


def test_role_and_worker_artifacts_never_collide(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    identities = (("train", 0), ("train", 1), ("validation", 0))
    trusted_descriptors: list[EpisodeSeedArtifactDescriptor] = []

    for ordinal, (role, worker_index) in enumerate(identities):
        wrapped = EpisodeSeedRecordingWrapper(
            ResetInfoEnv([seed_info(ordinal, 100 + ordinal, 200 + ordinal)]),
            workspace=workspace,
            role=role,
            worker_index=worker_index,
        )
        wrapped.reset()
        trusted_descriptors.append(wrapped.episode_seed_artifact_descriptor)
        wrapped.close()

    summaries = summarize_episode_seed_artifacts(
        workspace,
        expected_descriptors=trusted_descriptors,
    )
    assert [summary["path"] for summary in summaries] == [
        "episode_seeds/train-worker-000.jsonl",
        "episode_seeds/train-worker-001.jsonl",
        "episode_seeds/validation-worker-000.jsonl",
    ]
    assert [read_jsonl(workspace / str(summary["path"]))[0]["role"] for summary in summaries] == [
        "train",
        "train",
        "validation",
    ]


@pytest.mark.parametrize(
    "malformed_info",
    [
        {"metadrive_scenario_index": 1, "scenario_parameter_seed": 2},
        seed_info(True, 1, 2),
        seed_info(-1, 1, 2),
        seed_info(1, 1.5, 2),
        seed_info(1, 2, math.nan),
    ],
)
def test_malformed_reset_seed_info_fails_closed_without_a_record(
    tmp_path: Path,
    malformed_info: dict[str, Any],
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([malformed_info]),
        workspace=workspace,
        role="validation",
        worker_index=0,
    )

    with pytest.raises(ValueError, match="reset seed info"):
        wrapped.reset()

    artifact = workspace / "episode_seeds" / "validation-worker-000.jsonl"
    assert read_jsonl(artifact) == []
    wrapped.close()


def test_summary_rejects_malformed_or_unowned_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    artifact_dir = workspace / "episode_seeds"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "train-worker-000.jsonl"
    artifact.write_text('{"role":"train","worker_index":0}\n', encoding="utf-8")
    trusted_descriptor = descriptor_for_artifact(artifact, role="train", worker_index=0)

    with pytest.raises(ValueError, match="seed artifact"):
        summarize_episode_seed_artifacts(
            workspace,
            expected_descriptors=(trusted_descriptor,),
        )

    outside = tmp_path / "outside.jsonl"
    outside.write_text("foreign", encoding="utf-8")
    assert outside.read_text(encoding="utf-8") == "foreign"


@pytest.mark.parametrize(
    ("role", "worker_index", "message"),
    [
        ("unknown", 0, "role"),
        ("train", True, "worker_index"),
        ("train", -1, "worker_index"),
    ],
)
def test_recorder_rejects_invalid_environment_identity(
    tmp_path: Path,
    role: Any,
    worker_index: Any,
    message: str,
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match=message):
        EpisodeSeedRecordingWrapper(
            ResetInfoEnv([seed_info(1, 2, 3)]),
            workspace=workspace,
            role=role,
            worker_index=worker_index,
        )

    assert list(workspace.iterdir()) == []


def test_recorder_requires_an_existing_directory_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "not-a-directory"
    workspace.write_text("owned file", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="workspace"):
        EpisodeSeedRecordingWrapper(
            ResetInfoEnv([seed_info(1, 2, 3)]),
            workspace=workspace,
            role="train",
            worker_index=0,
        )

    assert workspace.read_text(encoding="utf-8") == "owned file"


def test_reset_forwards_nonempty_options_and_records_actual_info(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )

    wrapped.reset(options={"mode": "evaluation"})
    wrapped.close()

    artifact = workspace / "episode_seeds" / "train-worker-000.jsonl"
    assert read_jsonl(artifact)[0]["episode_rng_seed"] == 1


def test_summary_rejects_an_extra_file_in_owned_artifact_inventory(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )
    wrapped.reset()
    trusted_descriptor = wrapped.episode_seed_artifact_descriptor
    wrapped.close()
    extra = workspace / "episode_seeds" / "foreign.jsonl"
    extra.write_text("foreign", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory"):
        summarize_episode_seed_artifacts(
            workspace,
            expected_descriptors=(trusted_descriptor,),
        )

    assert extra.read_text(encoding="utf-8") == "foreign"


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("non_utf8", "unreadable"),
        ("incomplete", "incomplete final record"),
        ("bad_header", "header is malformed"),
        ("blank_record", "blank record"),
        ("bad_record_json", "record is malformed"),
        ("duplicate_record_field", "record is malformed"),
        ("bad_record_fields", "record fields are malformed"),
        ("bad_record_identity", "identity is malformed"),
    ],
)
def test_single_read_inventory_rejects_each_malformed_jsonl_boundary(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )
    wrapped.reset()
    trusted_descriptor = wrapped.episode_seed_artifact_descriptor
    wrapped.close()
    artifact = workspace / "episode_seeds" / "train-worker-000.jsonl"
    header = artifact.read_bytes().splitlines()[0]
    payloads = {
        "non_utf8": b"\xff\n",
        "incomplete": header,
        "bad_header": b"{\n",
        "blank_record": header + b"\n\n",
        "bad_record_json": header + b"\n{\n",
        "duplicate_record_field": header + b'\n{"episode_rng_seed":1,"episode_rng_seed":9,'
        b'"metadrive_scenario_index":2,"role":"train",'
        b'"scenario_parameter_seed":3,"worker_index":0}\n',
        "bad_record_fields": header + b'\n{"role":"train"}\n',
        "bad_record_identity": header + b'\n{"episode_rng_seed":1,"metadrive_scenario_index":2,'
        b'"role":"validation","scenario_parameter_seed":3,"worker_index":0}\n',
    }
    artifact.write_bytes(payloads[mode])

    with pytest.raises(ValueError, match=message):
        summarize_episode_seed_artifacts(
            workspace,
            expected_descriptors=(trusted_descriptor,),
        )


def test_closed_recorder_rejects_reset_without_retrying_resources(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    wrapped = EpisodeSeedRecordingWrapper(
        ResetInfoEnv([seed_info(1, 2, 3)]),
        workspace=workspace,
        role="train",
        worker_index=0,
    )

    wrapped.close()
    wrapped.close()

    with pytest.raises(RuntimeError, match="closed"):
        wrapped.reset()


def test_exclusive_writer_closes_descriptor_when_initial_write_makes_no_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    monkeypatch.setattr(episode_seed_module.os, "write", lambda descriptor, payload: 0)

    with pytest.raises(OSError, match="made no progress"):
        EpisodeSeedRecordingWrapper(
            ResetInfoEnv([seed_info(1, 2, 3)]),
            workspace=workspace,
            role="train",
            worker_index=0,
        )

    artifact = workspace / "episode_seeds" / "train-worker-000.jsonl"
    artifact.replace(workspace / "descriptor-was-closed.jsonl")


def test_writer_constructor_preserves_header_error_when_descriptor_close_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    real_close = episode_seed_module.os.close

    def fail_header_write(descriptor: int, payload: bytes) -> int:
        del descriptor, payload
        raise RuntimeError("header write failed")

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("descriptor close failed")

    monkeypatch.setattr(episode_seed_module.os, "write", fail_header_write)
    monkeypatch.setattr(episode_seed_module.os, "close", close_then_fail)

    with pytest.raises(RuntimeError, match="header write failed") as captured:
        EpisodeSeedRecordingWrapper(
            ResetInfoEnv([seed_info(1, 2, 3)]),
            workspace=workspace,
            role="train",
            worker_index=0,
        )

    assert any("descriptor close failed" in note for note in captured.value.__notes__)


def test_file_identity_rejects_an_unverifiable_inode() -> None:
    class UnverifiableStat:
        st_dev = 0
        st_ino = 0

    with pytest.raises(RuntimeError, match="no verifiable file identity"):
        episode_seed_module._file_identity(UnverifiableStat())  # type: ignore[arg-type]
