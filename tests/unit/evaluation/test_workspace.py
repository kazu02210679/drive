import json
from pathlib import Path

import pytest

from mad_driving.evaluation import workspace as workspace_module
from mad_driving.evaluation.workspace import EvaluationWorkspace


def test_workspace_stages_as_a_sibling_and_publishes_by_atomic_rename(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation"
    workspace = EvaluationWorkspace.stage(destination)
    artifact = workspace.path / "steps.jsonl"
    artifact.write_text("record\n", encoding="utf-8")
    workspace.write_manifest()

    published = workspace.publish()

    assert published == destination
    assert destination.joinpath("steps.jsonl").read_text(encoding="utf-8") == "record\n"
    assert not workspace.path.exists()


def test_workspace_never_overwrites_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation"
    destination.mkdir()
    destination.joinpath("sentinel").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        EvaluationWorkspace.stage(destination)

    assert destination.joinpath("sentinel").read_text(encoding="utf-8") == "keep"


def test_publish_rechecks_destination_collision_and_leaves_staging(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation"
    workspace = EvaluationWorkspace.stage(destination)
    destination.mkdir()

    with pytest.raises(FileExistsError):
        workspace.publish()

    assert workspace.path.is_dir()


def test_publish_does_not_replace_destination_created_at_rename_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "evaluation"
    workspace = EvaluationWorkspace.stage(destination)
    workspace.path.joinpath("steps.jsonl").write_text("record\n", encoding="utf-8")
    workspace.write_manifest()
    real_rename = workspace_module.rename_no_replace

    def occupy_then_rename(source: Path, target: Path) -> None:
        target.mkdir()
        target.joinpath("sentinel").write_text("foreign", encoding="utf-8")
        real_rename(source, target)

    monkeypatch.setattr(workspace_module, "rename_no_replace", occupy_then_rename)

    with pytest.raises(FileExistsError):
        workspace.publish()

    assert destination.joinpath("sentinel").read_text(encoding="utf-8") == "foreign"
    assert workspace.path.joinpath("steps.jsonl").read_text(encoding="utf-8") == "record\n"


def test_publish_rejects_missing_manifest_and_leaves_destination_absent(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation"
    workspace = EvaluationWorkspace.stage(destination)
    workspace.path.joinpath("steps.jsonl").write_text("record\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        workspace.publish()

    assert not destination.exists()


def test_manifest_is_deterministic_sorted_compact_json(tmp_path: Path) -> None:
    manifests: list[bytes] = []
    for name in ("first", "second"):
        workspace = EvaluationWorkspace.stage(tmp_path / name)
        workspace.path.joinpath("nested").mkdir()
        workspace.path.joinpath("z.jsonl").write_bytes(b"z\n")
        workspace.path.joinpath("nested", "a.jsonl").write_bytes(b"a\n")

        manifest_path = workspace.write_manifest()
        manifests.append(manifest_path.read_bytes())

    assert manifests[0] == manifests[1]
    assert manifests[0].endswith(b"\n") and not manifests[0].endswith(b"\n\n")
    payload = json.loads(manifests[0])
    assert [item["path"] for item in payload["artifacts"]] == ["nested/a.jsonl", "z.jsonl"]
    assert manifests[0] == (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def test_publish_rejects_file_modified_after_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation"
    workspace = EvaluationWorkspace.stage(destination)
    artifact = workspace.path / "steps.jsonl"
    artifact.write_text("before\n", encoding="utf-8")
    workspace.write_manifest()
    artifact.write_text("after!\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        workspace.publish()

    assert not destination.exists()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_publish_rejects_manifest_inventory_membership_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    destination = tmp_path / "evaluation"
    workspace = EvaluationWorkspace.stage(destination)
    workspace.path.joinpath("steps.jsonl").write_text("record\n", encoding="utf-8")
    manifest_path = workspace.write_manifest()
    payload = json.loads(manifest_path.read_bytes())
    if mutation == "missing":
        payload["artifacts"] = []
    else:
        payload["artifacts"].append({"path": "extra.jsonl", "size_bytes": 0, "sha256": "0" * 64})
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="manifest"):
        workspace.publish()

    assert not destination.exists()


@pytest.mark.parametrize("paths", [("../steps.jsonl",), ("steps.jsonl", "steps.jsonl")])
def test_publish_rejects_unsafe_or_duplicate_manifest_paths(
    tmp_path: Path, paths: tuple[str, ...]
) -> None:
    destination = tmp_path / "evaluation"
    workspace = EvaluationWorkspace.stage(destination)
    workspace.path.joinpath("steps.jsonl").write_text("record\n", encoding="utf-8")
    payload = {
        "artifacts": [{"path": path, "size_bytes": 7, "sha256": "0" * 64} for path in paths],
        "schema_version": 1,
    }
    workspace.path.joinpath("evaluation_manifest.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="manifest"):
        workspace.publish()

    assert not destination.exists()
