from pathlib import Path

import pytest

from mad_driving.evaluation.workspace import EvaluationWorkspace


def test_workspace_stages_as_a_sibling_and_publishes_by_atomic_rename(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation"
    workspace = EvaluationWorkspace.stage(destination)
    artifact = workspace.path / "steps.jsonl"
    artifact.write_text("record\n", encoding="utf-8")

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
