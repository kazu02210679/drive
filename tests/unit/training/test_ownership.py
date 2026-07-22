from pathlib import Path

import pytest

from mad_driving.atomic import rename_no_replace


def test_atomic_rename_no_replace_preserves_a_competing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    source.joinpath("owner").write_text("staged", encoding="utf-8")
    destination.mkdir()
    destination.joinpath("owner").write_text("foreign", encoding="utf-8")

    with pytest.raises(FileExistsError):
        rename_no_replace(source, destination)

    assert source.joinpath("owner").read_text(encoding="utf-8") == "staged"
    assert destination.joinpath("owner").read_text(encoding="utf-8") == "foreign"
