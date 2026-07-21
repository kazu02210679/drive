import dataclasses
import hashlib
from pathlib import Path

import pytest

from mad_driving.training import ResumeMetadata, RunMetadata, sha256_file
from mad_driving.training import metadata as metadata_module


def test_sha256_file_returns_lowercase_digest_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint.zip"
    original = b"checkpoint\x00payload"
    source.write_bytes(original)

    digest = sha256_file(source)

    assert digest == hashlib.sha256(original).hexdigest()
    assert len(digest) == 64
    assert digest == digest.lower()
    assert source.read_bytes() == original


def test_metadata_models_are_frozen_and_json_serializable() -> None:
    resume = ResumeMetadata(
        parent_checkpoint_path="C:/source/checkpoints/final_model.zip",
        parent_checkpoint_sha256="a" * 64,
        parent_run_dir="C:/source",
        parent_config={"training": {"seed": 42}},
        config_diff={"training.seed": {"parent": 42, "current": 43}},
        start_num_timesteps=12_500,
    )
    metadata = RunMetadata(
        resolved_config={"training": {"seed": 43}},
        resume=resume,
    )

    assert metadata.research_contract_version == 2
    assert metadata.observation_schema_version == 1
    assert metadata.observation_shape == (24,)
    assert metadata.action_schema_version == 1
    assert metadata.action_count == 4
    assert metadata.action_order == ("KEEP", "SLOW", "PREPARE_STOP", "STOP")
    assert dataclasses.asdict(metadata)["resume"]["start_num_timesteps"] == 12_500
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.action_count = 5  # type: ignore[misc]


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_metadata_write_failure_preserves_primary_error_and_cleans_sibling_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    destination = tmp_path / "run_metadata.json"
    metadata = RunMetadata(resolved_config={"seed": 42})
    expected_error: type[Exception] = TypeError
    expected_message = "not JSON serializable"
    if failure == "serialize":
        metadata = RunMetadata(resolved_config={"bad": object()})
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
