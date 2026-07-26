from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mad_driving.evaluation import bundle
from mad_driving.evaluation.models import Phase6PublicationPlan
from mad_driving.evaluation.training_metrics import TrainingMetricPoint
from mad_driving.evaluation.workspace import EvaluationWorkspace


def test_owned_tree_cleanup_rejects_unowned_root_and_non_directory(
    tmp_path: Path,
) -> None:
    owner = tmp_path / "owner"
    owner.mkdir()
    bundle._remove_owned_tree(owner / "missing", owner)

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(RuntimeError, match="unowned"):
        bundle._remove_owned_tree(outside, owner)
    with pytest.raises(RuntimeError, match="workspace root"):
        bundle._remove_owned_tree(owner, owner)

    regular = owner / "regular"
    regular.write_text("data", encoding="utf-8")
    with pytest.raises(RuntimeError, match="regular directory"):
        bundle._remove_owned_tree(regular, owner)

    child = owner / "child"
    child.mkdir()
    (child / "artifact").write_text("data", encoding="utf-8")
    bundle._remove_owned_tree(child, owner)
    assert not child.exists()


def test_workspace_cleanup_accepts_only_owned_sibling_staging_directory(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "evaluation"
    bundle._cleanup_workspace(None)
    missing = EvaluationWorkspace(destination, tmp_path / ".evaluation.staging-missing")
    bundle._cleanup_workspace(missing)

    wrong = tmp_path / "wrong"
    wrong.mkdir()
    with pytest.raises(RuntimeError, match="unproven"):
        bundle._cleanup_workspace(EvaluationWorkspace(destination, wrong))

    nested = tmp_path / "parent" / ".evaluation.staging-nested"
    nested.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="unproven"):
        bundle._cleanup_workspace(EvaluationWorkspace(destination, nested))

    owned = tmp_path / ".evaluation.staging-owned"
    owned.mkdir()
    bundle._cleanup_workspace(EvaluationWorkspace(destination, owned))
    assert not owned.exists()


def test_training_point_provenance_requires_one_run_per_method_seed(
    tmp_path: Path,
) -> None:
    del tmp_path

    plan = Phase6PublicationPlan.model_validate(
        {
            "plan_kind": "phase6_smoke",
            "evaluation_id": "test",
            "is_formal": False,
            "result_label": "SMOKE - NOT A RESEARCH RESULT",
            "app_config_path": "base.yaml",
            "method_overlays": [f"{index}.yaml" for index in range(7)],
            "max_episode_steps": 1,
            "episodes_per_case": 1,
            "test_seed_start": 20_000,
            "ppo_run_bindings": [
                {
                    "method_id": "proposed",
                    "policy_seed": 42,
                    "training_run_dir": "expected-run",
                    "checkpoint_path": "expected-run/model.zip",
                }
            ],
            "capture_episode_keys": [],
        }
    )
    points = (
        TrainingMetricPoint("first", "proposed", 42, 0, "policy_entropy", 1.0),
        TrainingMetricPoint("second", "proposed", 42, 1, "policy_entropy", 1.0),
    )
    with pytest.raises(ValueError, match="disagree"):
        bundle._validate_training_points(points, plan)
    with pytest.raises(ValueError, match="exactly"):
        bundle._validate_training_points(points[:1], plan)


def _manifest_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {"schema_version": 1},
        {"schema_version": True, "artifacts": []},
        {"schema_version": 1, "artifacts": {}},
        {"schema_version": 1, "artifacts": [1]},
        {
            "schema_version": 1,
            "artifacts": [{"path": "../bad", "size_bytes": 0, "sha256": "a" * 64}],
        },
        {
            "schema_version": 1,
            "artifacts": [{"path": "a", "size_bytes": True, "sha256": "a" * 64}],
        },
        {
            "schema_version": 1,
            "artifacts": [{"path": "a", "size_bytes": 0, "sha256": "bad"}],
        },
        {
            "schema_version": 1,
            "artifacts": [
                {"path": "a", "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
                {"path": "a", "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
            ],
        },
        {
            "schema_version": 1,
            "artifacts": [
                {"path": "b", "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
                {"path": "a", "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
            ],
        },
    ),
)
def test_workspace_manifest_validation_rejects_each_invalid_schema(
    tmp_path: Path,
    payload: object,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "evaluation_manifest.json").write_bytes(_manifest_bytes(payload))
    workspace = EvaluationWorkspace(tmp_path / "output", staging)
    with pytest.raises(ValueError):
        workspace._validate_manifest()


def test_workspace_manifest_rejects_noncanonical_inventory_and_tamper(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    manifest = staging / "evaluation_manifest.json"
    manifest.write_bytes(b'{\n  "artifacts": [],\n  "schema_version": 1\n}\n')
    workspace = EvaluationWorkspace(tmp_path / "output", staging)
    with pytest.raises(ValueError, match="canonical"):
        workspace._validate_manifest()

    manifest.write_bytes(
        _manifest_bytes(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "path": "missing",
                        "size_bytes": 0,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="inventory"):
        workspace._validate_manifest()

    artifact = staging / "artifact"
    artifact.write_bytes(b"actual")
    manifest.write_bytes(
        _manifest_bytes(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "path": "artifact",
                        "size_bytes": len(b"actual"),
                        "sha256": "a" * 64,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="verification"):
        workspace._validate_manifest()


def test_workspace_manifest_is_single_write_and_paths_are_strict(tmp_path: Path) -> None:
    workspace = EvaluationWorkspace.stage(tmp_path / "output")
    workspace.write_manifest()
    with pytest.raises(FileExistsError):
        workspace.write_manifest()
    from mad_driving.evaluation import workspace as workspace_module

    assert workspace_module._safe_manifest_path("artifact/path") is True
    for value in (None, "", "../bad", "/absolute", "C:/drive", "back\\slash", "./dot"):
        assert workspace_module._safe_manifest_path(value) is False
    with pytest.raises(ValueError, match="duplicate"):
        workspace_module._unique_json_object([("key", 1), ("key", 2)])
