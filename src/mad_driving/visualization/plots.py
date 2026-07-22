"""Deterministic headless plots from canonical verified metric CSV files."""

from __future__ import annotations

import csv
import io
import math
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Final

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mad_driving.atomic import rename_no_replace
from mad_driving.evaluation.compare import EVAL_METRICS_CSV_COLUMNS
from mad_driving.evaluation.training_metrics import TRAIN_METRICS_CSV_COLUMNS
from mad_driving.visualization import (
    METHOD_ORDER,
    PLOT_INVENTORY,
    SMOKE_RESULT_LABEL,
    _find_and_verify_bundle,
)

FIGURE_SIZE_INCHES: Final = (10.0, 6.0)
FIGURE_DPI: Final = 120
METHOD_COLORS: Final[Mapping[str, str]] = {
    "b0_rule": "#4C78A8",
    "b1_nominal": "#F58518",
    "b2_multi_no_review": "#54A24B",
    "proposed": "#E45756",
    "proposed_no_critic": "#72B7B2",
    "proposed_no_shield": "#B279A2",
    "proposed_no_hazard": "#FF9DA6",
}
_TRACK_ORDER: Final = {"decision": 0, "system": 1, "ablation": 2}
_PLOT_SPECS: Final = (
    ("collision_rate.png", "Safety: collision rate", ("collision",)),
    (
        "success_route_completion.png",
        "Efficiency: success and route completion",
        ("scenario_success", "final_route_completion"),
    ),
    (
        "unnecessary_braking.png",
        "Efficiency: unnecessary braking",
        ("unnecessary_braking_event_count", "unnecessary_stop_duration_s"),
    ),
    (
        "comfort.png",
        "Comfort",
        (
            "longitudinal_acceleration_rms_mps2",
            "maximum_deceleration_mps2",
            "longitudinal_jerk_rms_mps3",
        ),
    ),
    ("agent_disagreement.png", "Multi-Agent disagreement", ("agent_disagreement_rate",)),
)


def _read_csv(text: str, expected_columns: tuple[str, ...], label: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or tuple(reader.fieldnames) != expected_columns:
        raise ValueError(f"{label} must use the canonical fixed CSV columns")
    try:
        rows = list(reader)
    except csv.Error as error:
        raise ValueError(f"{label} is malformed CSV") from error
    if not rows:
        raise ValueError(f"{label} must contain at least one data row")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{label} contains malformed CSV cells")
    return rows


def _finite_cell(value: str, label: str) -> float | None:
    if value == "":
        return None
    try:
        numeric = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be numeric or empty") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite when available")
    return numeric


def _int_cell(value: str, label: str) -> int:
    try:
        numeric = int(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an integer") from error
    if numeric < 0:
        raise ValueError(f"{label} must be non-negative")
    return numeric


def _validate_common_rows(rows: Sequence[Mapping[str, str]], label: str) -> bool:
    labels = {row["result_label"] for row in rows}
    if not labels <= {"", SMOKE_RESULT_LABEL} or len(labels) != 1:
        raise ValueError(f"{label} has inconsistent result labels")
    for row in rows:
        if row["method_id"] not in METHOD_ORDER:
            raise ValueError(f"{label} contains an unknown method_id")
    return labels == {SMOKE_RESULT_LABEL}


def _new_figure(title: str) -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(figsize=FIGURE_SIZE_INCHES, dpi=FIGURE_DPI)
    axes.set_title(title)
    axes.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axes.set_axisbelow(True)
    return figure, axes


def _decorate(figure: Figure, *, smoke: bool, unavailable: bool) -> None:
    if unavailable:
        figure.text(
            0.985,
            0.02,
            "N/A",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#555555",
        )
    if smoke:
        figure.text(
            0.5,
            0.5,
            SMOKE_RESULT_LABEL,
            ha="center",
            va="center",
            fontsize=22,
            color="#B00020",
            alpha=0.22,
            rotation=28,
            weight="bold",
        )


def _save_figure(figure: Figure, destination: Path) -> None:
    output = Path(destination)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".png", dir=output.parent, delete=False
        ).name
    )
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=FIGURE_DPI,
            metadata={"Software": "mad-driving", "Creation Time": None},
        )
        rename_no_replace(temporary, output)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def write_learning_curve(train_metrics_csv: Path, output_png: Path) -> None:
    """Render episode-reward learning curves from verified canonical training metrics."""

    output = Path(output_png)
    if output.exists():
        raise FileExistsError(output)
    source = Path(train_metrics_csv)
    bundle = _find_and_verify_bundle(source)
    rows = _read_csv(bundle.read_text(source), TRAIN_METRICS_CSV_COLUMNS, "train_metrics.csv")
    smoke = _validate_common_rows(rows, "train_metrics.csv")
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    unavailable = False
    seen: set[tuple[str, int]] = set()
    for row in rows:
        timestep = _int_cell(row["timestep"], "training timestep")
        value = _finite_cell(row["value"], "training metric value")
        if value is None:
            unavailable = True
        if row["metric"] != "rollout/ep_rew_mean" or value is None:
            continue
        key = (row["method_id"], timestep)
        if key in seen:
            raise ValueError("train_metrics.csv contains duplicate learning-curve points")
        seen.add(key)
        series[row["method_id"]].append((timestep, value))
    figure, axes = _new_figure("Training episode reward")
    plotted = False
    for method_id in METHOD_ORDER:
        points = sorted(series.get(method_id, ()))
        if not points:
            continue
        plotted = True
        axes.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            label=method_id,
            color=METHOD_COLORS[method_id],
            linewidth=2.0,
            marker="o",
            markersize=3.5,
        )
    unavailable = unavailable or not plotted
    axes.set_xlabel("Training timestep")
    axes.set_ylabel("Mean episode reward")
    if plotted:
        axes.legend(loc="best", frameon=False)
    _decorate(figure, smoke=smoke, unavailable=unavailable)
    figure.tight_layout()
    _save_figure(figure, output)


def _ordered_groups(rows: Sequence[Mapping[str, str]]) -> tuple[tuple[str, str], ...]:
    groups = {(row["track"], row["method_id"]) for row in rows}
    for track, method_id in groups:
        if track not in _TRACK_ORDER:
            raise ValueError("eval_metrics.csv contains an unknown track")
        if method_id not in METHOD_ORDER:
            raise ValueError("eval_metrics.csv contains an unknown method_id")
    return tuple(
        sorted(groups, key=lambda item: (_TRACK_ORDER[item[0]], METHOD_ORDER.index(item[1])))
    )


def _metric_means(
    rows: Sequence[Mapping[str, str]], groups: Sequence[tuple[str, str]], metric: str
) -> list[float | None]:
    values: list[float | None] = []
    for track, method_id in groups:
        available = [
            value
            for row in rows
            if row["track"] == track
            and row["method_id"] == method_id
            and (value := _finite_cell(row[metric], f"eval metric {metric}")) is not None
        ]
        values.append(fmean(available) if available else None)
    return values


def _render_metric_plot(
    rows: Sequence[Mapping[str, str]],
    groups: Sequence[tuple[str, str]],
    *,
    filename: str,
    title: str,
    metrics: Sequence[str],
    smoke: bool,
    destination: Path,
) -> None:
    figure, axes = _new_figure(title)
    width = 0.8 / len(metrics)
    unavailable = False
    for metric_index, metric in enumerate(metrics):
        values = _metric_means(rows, groups, metric)
        offset = (metric_index - (len(metrics) - 1) / 2) * width
        for group_index, ((_, method_id), value) in enumerate(zip(groups, values, strict=True)):
            x_position = group_index + offset
            if value is None:
                unavailable = True
                axes.bar(
                    x_position,
                    0.0,
                    width=width,
                    color="none",
                    edgecolor=METHOD_COLORS[method_id],
                    hatch="//",
                )
                axes.text(x_position, 0.0, "N/A", ha="center", va="bottom", fontsize=7, rotation=90)
            else:
                axes.bar(
                    x_position,
                    value,
                    width=width,
                    color=METHOD_COLORS[method_id],
                    alpha=0.82,
                    label=metric if group_index == 0 else None,
                )
    axes.set_xticks(
        range(len(groups)),
        [f"{track}\n{method_id}" for track, method_id in groups],
        rotation=30,
        ha="right",
    )
    axes.set_ylabel("Mean canonical metric value")
    if len(metrics) > 1:
        axes.legend(loc="best", frameon=False)
    _decorate(figure, smoke=smoke, unavailable=unavailable)
    figure.tight_layout()
    _save_figure(figure, destination / filename)


def write_safety_efficiency_plots(eval_metrics_csv: Path, output_dir: Path) -> tuple[Path, ...]:
    """Render the fixed five evaluation figures from verified canonical episode metrics."""

    destination = Path(output_dir)
    outputs = tuple(destination / spec[0] for spec in _PLOT_SPECS)
    existing = next((path for path in outputs if path.exists()), None)
    if existing is not None:
        raise FileExistsError(existing)
    source = Path(eval_metrics_csv)
    bundle = _find_and_verify_bundle(source)
    rows = _read_csv(bundle.read_text(source), EVAL_METRICS_CSV_COLUMNS, "eval_metrics.csv")
    smoke = _validate_common_rows(rows, "eval_metrics.csv")
    groups = _ordered_groups(rows)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, title, metrics in _PLOT_SPECS:
        _render_metric_plot(
            rows,
            groups,
            filename=filename,
            title=title,
            metrics=metrics,
            smoke=smoke,
            destination=destination,
        )
    return outputs


__all__ = [
    "FIGURE_DPI",
    "FIGURE_SIZE_INCHES",
    "METHOD_COLORS",
    "METHOD_ORDER",
    "PLOT_INVENTORY",
    "write_learning_curve",
    "write_safety_efficiency_plots",
]
