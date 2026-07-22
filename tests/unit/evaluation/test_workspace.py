import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

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


def test_nested_regular_directories_manifest_and_publish(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation"
    workspace = EvaluationWorkspace.stage(destination)
    workspace.path.joinpath("nested", "deeper").mkdir(parents=True)
    workspace.path.joinpath("root.jsonl").write_bytes(b"root\n")
    workspace.path.joinpath("nested", "deeper", "step.jsonl").write_bytes(b"step\n")

    manifest_path = workspace.write_manifest()
    payload = json.loads(manifest_path.read_bytes())
    published = workspace.publish()

    assert [item["path"] for item in payload["artifacts"]] == [
        "nested/deeper/step.jsonl",
        "root.jsonl",
    ]
    assert published.joinpath("nested", "deeper", "step.jsonl").read_bytes() == b"step\n"


@pytest.mark.parametrize("link_kind", ["directory", "dangling"])
@pytest.mark.parametrize("after_manifest", [False, True])
def test_manifest_rejects_symlinks_before_regular_file_filtering(
    tmp_path: Path, link_kind: str, after_manifest: bool
) -> None:
    destination = tmp_path / "evaluation"
    workspace = EvaluationWorkspace.stage(destination)
    workspace.path.joinpath("steps.jsonl").write_bytes(b"step\n")
    if after_manifest:
        workspace.write_manifest()
    link = workspace.path / f"{link_kind}-link"
    target = tmp_path / "real-directory" if link_kind == "directory" else tmp_path / "missing"
    if link_kind == "directory":
        target.mkdir()
    try:
        os.symlink(target, link, target_is_directory=link_kind == "directory")
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(ValueError, match="symbolic|reparse|link"):
        if after_manifest:
            workspace.publish()
        else:
            workspace.write_manifest()

    assert not destination.exists()


class FakeReparseDirectoryEntry:
    name = "junction"
    path = "junction"

    @staticmethod
    def is_symlink() -> bool:
        return False

    @staticmethod
    def stat(*, follow_symlinks: bool = True) -> SimpleNamespace:
        assert follow_symlinks is False
        return SimpleNamespace(
            st_mode=stat.S_IFDIR,
            st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
        )


def test_windows_junction_reparse_entry_is_rejected_without_following() -> None:
    with pytest.raises(ValueError, match="reparse"):
        workspace_module._validated_entry_kind(FakeReparseDirectoryEntry())


class FakeSpecialEntry(FakeReparseDirectoryEntry):
    name = "named-pipe"
    path = "named-pipe"

    @staticmethod
    def stat(*, follow_symlinks: bool = True) -> SimpleNamespace:
        assert follow_symlinks is False
        return SimpleNamespace(st_mode=stat.S_IFIFO, st_file_attributes=0)


def test_non_regular_non_directory_entry_is_rejected() -> None:
    with pytest.raises(ValueError, match="regular file or directory"):
        workspace_module._validated_entry_kind(FakeSpecialEntry())


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
