import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from mad_driving.training import ResumeMetadata, RunMetadata, sha256_file
from mad_driving.training import metadata as metadata_module


def curriculum_summary(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "path": "curriculum_state.yaml",
        "sha256": "e" * 64,
        "level": 2,
        "consecutive_passes": 1,
        "evaluations": 7,
    }
    values.update(overrides)
    return values


def test_sha256_file_returns_lowercase_digest_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint.zip"
    original = b"checkpoint\x00payload"
    source.write_bytes(original)

    digest = sha256_file(source)

    assert digest == hashlib.sha256(original).hexdigest()
    assert len(digest) == 64
    assert digest == digest.lower()
    assert source.read_bytes() == original


def test_metadata_models_are_frozen_and_json_serializable(tmp_path: Path) -> None:
    parent_run = (tmp_path / "source").resolve()
    parent_checkpoint = (parent_run / "checkpoints" / "final_model.zip").resolve()
    resume = ResumeMetadata(
        parent_checkpoint_path=str(parent_checkpoint),
        parent_checkpoint_sha256="a" * 64,
        parent_run_dir=str(parent_run),
        parent_config={"training": {"seed": 42}},
        config_diff={"training.seed": {"parent": 42, "current": 43}},
        start_num_timesteps=12_500,
    )
    metadata = RunMetadata(
        resolved_config={"training": {"seed": 43}},
        resume=resume,
        curriculum_state=curriculum_summary(),
    )

    assert metadata.research_contract_version == 6
    assert metadata.observation_schema_version == 1
    assert metadata.observation_shape == (24,)
    assert metadata.observation_dtype == "float32"
    assert metadata.action_schema_version == 1
    assert metadata.action_count == 4
    assert metadata.action_order == ("KEEP", "SLOW", "PREPARE_STOP", "STOP")
    assert metadata.curriculum_state == curriculum_summary()
    assert dataclasses.asdict(metadata)["resume"]["start_num_timesteps"] == 12_500
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.action_count = 5  # type: ignore[misc]


def test_contract_5_run_metadata_requires_non_null_curriculum_state() -> None:
    with pytest.raises(ValueError, match="curriculum_state"):
        RunMetadata(resolved_config={"seed": 42}, curriculum_state=None)


def test_metadata_recursively_detaches_and_freezes_caller_owned_values(tmp_path: Path) -> None:
    resolved_config: dict[str, Any] = {
        "training": {"seed": 42, "layers": [24, 16]},
        "enabled": True,
    }
    parent_config: dict[str, Any] = {"training": {"seed": 41}}
    config_diff: dict[str, Any] = {"training.seed": {"parent": 41, "current": 42}}
    parent_run = (tmp_path / "source").resolve()
    resume = ResumeMetadata(
        parent_checkpoint_path=str((parent_run / "checkpoints" / "final.zip").resolve()),
        parent_checkpoint_sha256="b" * 64,
        parent_run_dir=str(parent_run),
        parent_config=parent_config,
        config_diff=config_diff,
        start_num_timesteps=12_500,
    )
    metadata = RunMetadata(
        resolved_config=resolved_config,
        curriculum_state=curriculum_summary(),
        resume=resume,
    )

    resolved_config["training"]["seed"] = 999
    resolved_config["training"]["layers"].append(8)
    parent_config["training"]["seed"] = 999
    config_diff["training.seed"]["current"] = 999

    assert metadata.resolved_config["training"]["seed"] == 42
    assert metadata.resolved_config["training"]["layers"] == (24, 16)
    assert metadata.resume is not None
    assert metadata.resume.parent_config["training"]["seed"] == 41
    assert metadata.resume.config_diff["training.seed"]["current"] == 42
    with pytest.raises(TypeError):
        metadata.resolved_config["training"]["seed"] = 0  # type: ignore[index]
    with pytest.raises(AttributeError):
        metadata.resolved_config["training"]["layers"].append(8)


def test_frozen_json_objects_reject_attribute_and_storage_reassignment() -> None:
    metadata = RunMetadata(
        resolved_config={"nested": {"value": 7}},
        curriculum_state=curriculum_summary(),
    )
    root = metadata.resolved_config
    nested = root["nested"]

    with pytest.raises(AttributeError):
        root._items = (("changed", True),)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        setattr(root, "extra", "foreign")  # noqa: B010 - explicit adversarial API test
    with pytest.raises(AttributeError):
        nested._items = (("value", 99),)  # type: ignore[attr-defined]

    assert root == {"nested": {"value": 7}}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"research_contract_version": True}, "research_contract_version"),
        ({"research_contract_version": 4}, "research_contract_version"),
        ({"research_contract_version": 3}, "research_contract_version"),
        ({"research_contract_version": 2}, "research_contract_version"),
        ({"research_contract_version": 1}, "research_contract_version"),
        ({"observation_schema_version": 2}, "observation_schema_version"),
        ({"observation_shape": (25,)}, "observation_shape"),
        ({"observation_shape": (24.0,)}, "observation_shape"),
        ({"observation_shape": (True,)}, "observation_shape"),
        ({"observation_shape": ("24",)}, "observation_shape"),
        ({"observation_dtype": "float64"}, "observation_dtype"),
        ({"action_schema_version": False}, "action_schema_version"),
        ({"action_count": True}, "action_count"),
        ({"action_count": 5}, "action_count"),
        ({"action_order": ("STOP", "SLOW", "PREPARE_STOP", "KEEP")}, "action_order"),
        ({"resolved_config": {"value": math.nan}}, "finite"),
        ({"resolved_config": {"value": math.inf}}, "finite"),
        ({"resolved_config": {"value": object()}}, "JSON"),
        ({"resolved_config": {1: "non-string"}}, "string key"),
    ],
)
def test_run_metadata_rejects_invalid_public_constructor_fields(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "resolved_config": {"seed": 42},
        "curriculum_state": curriculum_summary(),
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        RunMetadata(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parent_checkpoint_path": ""}, "non-empty"),
        ({"parent_run_dir": ""}, "non-empty"),
        ({"parent_checkpoint_path": 7}, "non-empty"),
        ({"parent_checkpoint_sha256": "A" * 64}, "SHA-256"),
        ({"parent_checkpoint_sha256": "a" * 63}, "SHA-256"),
        ({"parent_checkpoint_sha256": "z" * 64}, "SHA-256"),
        ({"start_num_timesteps": True}, "start_num_timesteps"),
        ({"start_num_timesteps": -1}, "start_num_timesteps"),
        ({"start_num_timesteps": 1.5}, "start_num_timesteps"),
        ({"parent_config": {"value": -math.inf}}, "finite"),
        ({"config_diff": {"value": object()}}, "JSON"),
    ],
)
def test_resume_metadata_rejects_invalid_public_constructor_fields(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    parent_run = (tmp_path / "source").resolve()
    values: dict[str, object] = {
        "parent_checkpoint_path": str((parent_run / "checkpoints" / "final.zip").resolve()),
        "parent_checkpoint_sha256": "c" * 64,
        "parent_run_dir": str(parent_run),
        "parent_config": {"seed": 42},
        "config_diff": {},
        "start_num_timesteps": 0,
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        ResumeMetadata(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("checkpoint_path", "run_dir"),
    [
        pytest.param(
            r"C:\research\foreign-host\checkpoints\final_model.zip",
            r"C:\research\foreign-host",
            id="windows-style-on-posix-compatible-parser",
        ),
        pytest.param(
            "/mnt/research/foreign-host/checkpoints/final_model.zip",
            "/mnt/research/foreign-host",
            id="posix-style-on-windows-compatible-parser",
        ),
    ],
)
def test_historical_resume_provenance_is_parsed_without_current_host_path_semantics(
    checkpoint_path: str,
    run_dir: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PathUseWouldDereferenceHistoricalProvenance:
        def __init__(self, value: object) -> None:
            raise AssertionError(f"historical provenance was reparsed as a local path: {value}")

    monkeypatch.setattr(metadata_module, "Path", PathUseWouldDereferenceHistoricalProvenance)

    parsed = metadata_module._parse_resume_metadata(
        {
            "parent_checkpoint_path": checkpoint_path,
            "parent_checkpoint_sha256": "d" * 64,
            "parent_run_dir": run_dir,
            "parent_config": {"seed": 42},
            "config_diff": {},
            "start_num_timesteps": 500,
        }
    )

    assert parsed is not None
    assert parsed.parent_checkpoint_path == checkpoint_path
    assert parsed.parent_run_dir == run_dir


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_metadata_write_failure_preserves_primary_error_and_cleans_sibling_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    destination = tmp_path / "run_metadata.json"
    metadata = RunMetadata(
        resolved_config={"seed": 42},
        curriculum_state=curriculum_summary(),
    )
    expected_error: type[Exception] = TypeError
    expected_message = "not JSON serializable"
    if failure == "serialize":
        object.__setattr__(metadata, "resolved_config", {"bad": object()})
        expected_error = ValueError
        expected_message = "JSON safe"
    else:
        expected_error = OSError
        expected_message = "replace failed"

        def fail_replace(source: Path, target: Path) -> None:
            del source, target
            raise OSError("replace failed")

        monkeypatch.setattr(metadata_module.os, "replace", fail_replace)

    with pytest.raises(expected_error, match=expected_message):
        metadata_module.write_run_metadata(metadata, destination)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_metadata_writer_rejects_non_finite_nested_value_without_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "run_metadata.json"
    valid = RunMetadata(
        resolved_config={"seed": 42},
        curriculum_state=curriculum_summary(),
    )
    object.__setattr__(valid, "resolved_config", {"nested": [float("nan")]})

    with pytest.raises(ValueError, match="finite"):
        metadata_module.write_run_metadata(valid, destination)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_contract_6_metadata_without_seed_artifacts_remains_loadable(tmp_path: Path) -> None:
    destination = tmp_path / "run_metadata.json"
    metadata_module.write_run_metadata(
        RunMetadata(
            resolved_config={"seed": 42},
            curriculum_state=curriculum_summary(),
        ),
        destination,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload.pop("episode_seed_artifacts")
    destination.write_text(json.dumps(payload), encoding="utf-8")

    loaded = metadata_module._load_run_metadata(destination)

    assert loaded.research_contract_version == 6
    assert loaded.episode_seed_artifacts == ()


def test_metadata_writes_curriculum_state_values_and_sha256(tmp_path: Path) -> None:
    destination = tmp_path / "run_metadata.json"
    summary = curriculum_summary()

    metadata_module.write_run_metadata(
        RunMetadata(
            resolved_config={"seed": 42},
            curriculum_state=summary,
        ),
        destination,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["curriculum_state"] == summary


def test_run_metadata_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    destination = tmp_path / "run_metadata.json"
    metadata_module.write_run_metadata(
        RunMetadata(
            resolved_config={"seed": 42},
            curriculum_state=curriculum_summary(),
        ),
        destination,
    )
    text = destination.read_text(encoding="utf-8")
    text = text.replace(
        '  "research_contract_version": 6,',
        '  "research_contract_version": 6,\n  "research_contract_version": 6,',
        1,
    )
    destination.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        metadata_module._load_run_metadata(destination)


@pytest.mark.parametrize(
    "summary",
    [
        curriculum_summary(path="../outside.yaml"),
        curriculum_summary(sha256="A" * 64),
        curriculum_summary(level=True),
        curriculum_summary(level=4),
        curriculum_summary(consecutive_passes=8),
        {**curriculum_summary(), "extra": True},
    ],
)
def test_metadata_rejects_malformed_curriculum_state_summary(
    summary: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="curriculum_state"):
        RunMetadata(
            resolved_config={"seed": 42},
            curriculum_state=summary,
        )


@pytest.mark.parametrize(
    "summary",
    [
        {
            "path": "../outside.jsonl",
            "record_count": 1,
            "role": "train",
            "schema_version": 1,
            "sha256": "a" * 64,
            "worker_index": 0,
        },
        {
            "path": "episode_seeds/train-worker-000.jsonl",
            "record_count": -1,
            "role": "train",
            "schema_version": 1,
            "sha256": "a" * 64,
            "worker_index": 0,
        },
        {
            "path": "episode_seeds/train-worker-000.jsonl",
            "record_count": 1,
            "role": "train",
            "schema_version": 2,
            "sha256": "a" * 64,
            "worker_index": 0,
        },
    ],
)
def test_metadata_rejects_malformed_episode_seed_artifact_summary(
    summary: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="episode_seed_artifacts"):
        RunMetadata(
            resolved_config={"seed": 42},
            curriculum_state=curriculum_summary(),
            episode_seed_artifacts=(summary,),
        )
