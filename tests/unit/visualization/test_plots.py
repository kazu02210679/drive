from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

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


def test_png_outputs_repeat_byte_for_byte_across_independent_bundles(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundles = (bundle_factory(), bundle_factory())
    destinations = (tmp_path / "repeat-a", tmp_path / "repeat-b")

    for bundle, destination in zip(bundles, destinations, strict=True):
        write_learning_curve(
            bundle / "metrics" / "train_metrics.csv",
            destination / "learning_curve.png",
        )
        write_safety_efficiency_plots(
            bundle / "metrics" / "eval_metrics.csv",
            destination,
        )

    for filename in PLOT_INVENTORY:
        assert (destinations[0] / filename).read_bytes() == (
            destinations[1] / filename
        ).read_bytes()


def test_plots_emit_actual_fixed_legend_and_tick_label_order(
    bundle_factory: Callable[[], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    legends: list[tuple[str, ...]] = []
    tick_labels: list[tuple[str, ...]] = []
    original_legend = Axes.legend
    original_set_xticks = Axes.set_xticks

    def capture_legend(self: Axes, *args: object, **kwargs: object):
        legends.append(tuple(self.get_legend_handles_labels()[1]))
        return original_legend(self, *args, **kwargs)

    def capture_ticks(
        self: Axes,
        ticks: object,
        labels: object = None,
        *args: object,
        **kwargs: object,
    ):
        if labels is not None:
            tick_labels.append(tuple(str(label) for label in labels))
        return original_set_xticks(self, ticks, labels, *args, **kwargs)

    monkeypatch.setattr(Axes, "legend", capture_legend)
    monkeypatch.setattr(Axes, "set_xticks", capture_ticks)
    bundle = bundle_factory()
    write_learning_curve(
        bundle / "metrics" / "train_metrics.csv",
        tmp_path / "ordered" / "learning_curve.png",
    )
    write_safety_efficiency_plots(
        bundle / "metrics" / "eval_metrics.csv",
        tmp_path / "ordered",
    )

    assert legends == [
        ("proposed",),
        ("scenario_success", "final_route_completion"),
        ("unnecessary_braking_event_count", "unnecessary_stop_duration_s"),
        (
            "longitudinal_acceleration_rms_mps2",
            "maximum_deceleration_mps2",
            "longitudinal_jerk_rms_mps3",
        ),
    ]
    expected = (
        "decision\nb1_nominal",
        "system\nb0_rule",
        "ablation\nproposed",
    )
    assert tick_labels == [expected] * 5


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


def test_verified_consumers_use_handle_bytes_not_path_reopens(
    bundle_factory: Callable[[], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = bundle_factory()
    trace = next(bundle.glob("episodes/**/*_trace.jsonl"))
    frames = trace.parent / "episode_20001_frames"

    def forbidden_read(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("verified consumers must not reopen artifacts through Path")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    monkeypatch.setattr(Path, "read_text", forbidden_read)

    write_learning_curve(
        bundle / "metrics" / "train_metrics.csv",
        tmp_path / "handle-only.png",
    )
    from mad_driving.visualization.overlay import write_episode_gif
    from mad_driving.visualization.report import write_markdown_report

    write_episode_gif(trace, frames, tmp_path / "handle-only.gif")
    write_markdown_report(bundle, tmp_path / "handle-only.md")


def test_verified_handle_read_rejects_identity_change_during_read(
    bundle_factory: Callable[[], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = bundle_factory()
    import mad_driving.visualization as visualization

    original_fstat = os.fstat
    calls_by_fd: dict[int, int] = {}

    def drifting_fstat(fd: int) -> SimpleNamespace:
        metadata = original_fstat(fd)
        calls_by_fd[fd] = calls_by_fd.get(fd, 0) + 1
        return SimpleNamespace(
            st_mode=metadata.st_mode,
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_size=metadata.st_size,
            st_mtime_ns=metadata.st_mtime_ns + (calls_by_fd[fd] - 1),
            st_file_attributes=getattr(metadata, "st_file_attributes", 0),
        )

    monkeypatch.setattr(visualization.os, "fstat", drifting_fstat)
    output = tmp_path / "drift.png"
    with pytest.raises(ValueError, match="changed|identity|verification"):
        write_learning_curve(bundle / "metrics" / "train_metrics.csv", output)
    assert not output.exists()
