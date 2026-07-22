from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import imageio.v3 as iio
import pytest

from mad_driving.evaluation.workspace import EvaluationWorkspace
from mad_driving.visualization.plots import (
    FIGURE_DPI,
    FIGURE_SIZE_INCHES,
    METHOD_ORDER,
    PLOT_INVENTORY,
    write_learning_curve,
    write_safety_efficiency_plots,
)


def test_writes_fixed_deterministic_png_inventory_from_canonical_csvs(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    first_bundle = bundle_factory()
    first_output = tmp_path / "first"

    learning = first_output / "learning_curve.png"
    write_learning_curve(first_bundle / "metrics" / "train_metrics.csv", learning)
    remaining = write_safety_efficiency_plots(
        first_bundle / "metrics" / "eval_metrics.csv", first_output
    )

    assert (learning.name, *(path.name for path in remaining)) == PLOT_INVENTORY
    assert METHOD_ORDER == (
        "b0_rule",
        "b1_nominal",
        "b2_multi_no_review",
        "proposed",
        "proposed_no_critic",
        "proposed_no_shield",
        "proposed_no_hazard",
    )
    expected_shape = (
        round(FIGURE_SIZE_INCHES[1] * FIGURE_DPI),
        round(FIGURE_SIZE_INCHES[0] * FIGURE_DPI),
    )
    for output in (learning, *remaining):
        image = iio.imread(output)
        assert image.shape[:2] == expected_shape
        assert image.size > 0


def test_unavailable_values_render_as_na_and_smoke_watermark_is_visible(
    bundle_factory: Callable[[], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = bundle_factory()
    visible_text: list[str] = []

    from matplotlib.figure import Figure

    original = Figure.text

    def capture(self: Figure, x: float, y: float, text: str, *args: object, **kwargs: object):
        visible_text.append(text)
        return original(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(Figure, "text", capture)
    write_learning_curve(bundle / "metrics" / "train_metrics.csv", tmp_path / "curve.png")
    write_safety_efficiency_plots(bundle / "metrics" / "eval_metrics.csv", tmp_path / "plots")

    assert "SMOKE - NOT A RESEARCH RESULT" in visible_text
    assert "N/A" in visible_text
    assert not any("nan" in label.lower() for label in visible_text)


def test_plot_reader_rejects_tampered_and_undeclared_csv_before_output(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()
    metrics = bundle / "metrics" / "train_metrics.csv"
    metrics.write_bytes(metrics.read_bytes() + b"tampered")
    output = tmp_path / "tampered.png"
    with pytest.raises(ValueError, match="manifest|verification|integrity"):
        write_learning_curve(metrics, output)
    assert not output.exists()

    clean = bundle_factory()
    undeclared = clean / "metrics" / "undeclared.csv"
    undeclared.write_bytes((clean / "metrics" / "train_metrics.csv").read_bytes())
    with pytest.raises(ValueError, match="manifest|undeclared|inventory"):
        write_learning_curve(undeclared, tmp_path / "undeclared.png")


@pytest.mark.parametrize(
    "unsafe_path",
    ("../outside.csv", "/absolute.csv", "C:/absolute.csv", "metrics\\train_metrics.csv"),
)
def test_plot_reader_rejects_unsafe_manifest_paths(
    bundle_factory: Callable[[], Path], tmp_path: Path, unsafe_path: str
) -> None:
    bundle = bundle_factory()
    manifest = bundle / "evaluation_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"][0]["path"] = unsafe_path
    manifest.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest|unsafe"):
        write_learning_curve(bundle / "metrics" / "train_metrics.csv", tmp_path / "unsafe.png")


def test_plot_writers_never_overwrite_existing_outputs(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()
    existing = tmp_path / "learning_curve.png"
    existing.write_bytes(b"owned")
    with pytest.raises(FileExistsError):
        write_learning_curve(bundle / "metrics" / "train_metrics.csv", existing)
    assert existing.read_bytes() == b"owned"

    output_dir = tmp_path / "plots"
    output_dir.mkdir()
    collision = output_dir / "collision_rate.png"
    collision.write_bytes(b"owned")
    with pytest.raises(FileExistsError):
        write_safety_efficiency_plots(bundle / "metrics" / "eval_metrics.csv", output_dir)
    assert collision.read_bytes() == b"owned"


def test_visualization_integrity_rejects_reparse_and_special_entries() -> None:
    from types import SimpleNamespace

    from mad_driving.visualization import _validated_entry_kind

    class FakeEntry:
        name = "entry"
        path = "entry"

        @staticmethod
        def is_symlink() -> bool:
            return False

        @staticmethod
        def stat(*, follow_symlinks: bool = True) -> SimpleNamespace:
            assert follow_symlinks is False
            return SimpleNamespace(st_mode=0, st_file_attributes=0x400)

    with pytest.raises(ValueError, match="reparse"):
        _validated_entry_kind(FakeEntry())

    class SpecialEntry(FakeEntry):
        @staticmethod
        def stat(*, follow_symlinks: bool = True) -> SimpleNamespace:
            assert follow_symlinks is False
            return SimpleNamespace(st_mode=0x1000, st_file_attributes=0)

    with pytest.raises(ValueError, match="regular file or directory"):
        _validated_entry_kind(SpecialEntry())


def test_visualization_reader_rejects_symlink_entries_when_supported(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = bundle / "linked.txt"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(ValueError, match="symbolic|reparse|link"):
        write_learning_curve(bundle / "metrics" / "train_metrics.csv", tmp_path / "linked.png")


def test_bundle_manifest_used_by_tests_has_exact_inventory(
    bundle_factory: Callable[[], Path],
) -> None:
    bundle = bundle_factory()
    EvaluationWorkspace(destination=bundle, path=bundle)._validate_manifest()
