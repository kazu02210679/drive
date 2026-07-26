from __future__ import annotations

import csv
import io
from collections.abc import Callable
from pathlib import Path

import pytest

from mad_driving.evaluation.compare import COMPARISON_CSV_COLUMNS, EVAL_METRICS_CSV_COLUMNS
from mad_driving.evaluation.training_metrics import TRAIN_METRICS_CSV_COLUMNS
from mad_driving.visualization import report


def _mutated_csv(
    path: Path,
    mutate: Callable[[list[dict[str, str]]], None],
) -> bytes:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames is not None
        columns = tuple(reader.fieldnames)
        rows = list(reader)
    mutate(rows)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def test_csv_scalar_boundaries_reject_noncanonical_values() -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        report._raw_csv_rows(b"\xff", "data.csv", ("field",))
    with pytest.raises(ValueError, match="columns"):
        report._raw_csv_rows(b"wrong\nvalue\n", "data.csv", ("field",))
    with pytest.raises(ValueError, match="complete"):
        report._raw_csv_rows(b"field,extra\nvalue\n", "data.csv", ("field", "extra"))
    for value in ("", "01", "-1", "1.0"):
        with pytest.raises(ValueError, match="canonical integer"):
            report._integer(value, "value")
    with pytest.raises(ValueError, match="at least"):
        report._integer("0", "value", minimum=1)
    with pytest.raises(ValueError, match="boolean"):
        report._boolean("true", "value")
    with pytest.raises(ValueError, match="numeric"):
        report._number("value", "number")
    with pytest.raises(ValueError, match="finite"):
        report._number("nan", "number")
    with pytest.raises(ValueError, match="numeric"):
        report._number("", "number", allow_empty=False)
    with pytest.raises(ValueError, match="result_label"):
        report._row_formality("unknown", None, "row")
    with pytest.raises(ValueError, match="inconsistent"):
        report._row_formality("", "False", "row")
    with pytest.raises(ValueError, match="mixes"):
        report._require_one_formality((True, False), "rows")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_id", " bad"),
        ("method_id", "b0_rule"),
        ("policy_seed", "01"),
        ("timestep", "-1"),
        ("metric", "unknown"),
        ("value", "nan"),
    ),
)
def test_train_metrics_parser_rejects_each_noncanonical_domain(
    bundle_factory: Callable[[], Path],
    field: str,
    value: str,
) -> None:
    bundle = bundle_factory()

    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0][field] = value

    payload = _mutated_csv(bundle / "metrics/train_metrics.csv", mutate)
    with pytest.raises(ValueError):
        report._parse_train_metrics_csv(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("track", "unknown"),
        ("method_id", "b0_rule"),
        ("policy_seed", ""),
        ("case_id", "unknown"),
        ("test_seed", "19999"),
        ("record_schema_version", "2"),
        ("research_contract_version", "6"),
        ("checkpoint_path", ""),
        ("checkpoint_sha256", "bad"),
        ("policy_kind", "rule"),
        ("specialist_ids", "bad"),
        ("critic_enabled", "True"),
        ("shield_mode", "off"),
        ("scenario_id", "cut_in"),
        ("difficulty_level", "3"),
        ("metadrive_scenario_index", "-1"),
        ("scenario_selection_seed", "-1"),
        ("scenario_parameter_seed", "-1"),
    ),
)
def test_eval_metrics_parser_rejects_each_provenance_mismatch(
    bundle_factory: Callable[[], Path],
    field: str,
    value: str,
) -> None:
    bundle = bundle_factory()

    def mutate(rows: list[dict[str, str]]) -> None:
        row = next(item for item in rows if item["method_id"] == "b1_nominal")
        row[field] = value

    payload = _mutated_csv(bundle / "metrics/eval_metrics.csv", mutate)
    with pytest.raises(ValueError):
        report._parse_eval_metrics_csv(payload)


def test_eval_metrics_parser_rejects_duplicate_episode_identity(
    bundle_factory: Callable[[], Path],
) -> None:
    bundle = bundle_factory()

    def duplicate(rows: list[dict[str, str]]) -> None:
        rows.insert(1, dict(rows[0]))

    payload = _mutated_csv(bundle / "metrics/eval_metrics.csv", duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        report._parse_eval_metrics_csv(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("track", "unknown"),
        ("case_id", "unknown"),
        ("metric", "unknown"),
        ("physical_episode_count", "0"),
        ("policy_replicate_count", "0"),
        ("policy_seed_stdev", "-1"),
    ),
)
def test_comparison_parser_rejects_invalid_domains(
    bundle_factory: Callable[[], Path],
    field: str,
    value: str,
) -> None:
    bundle = bundle_factory()

    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0][field] = value

    payload = _mutated_csv(bundle / "metrics/comparison.csv", mutate)
    with pytest.raises(ValueError):
        report._parse_comparison_csv(payload)


@pytest.mark.parametrize(
    ("metric", "mean", "stdev"),
    (
        ("collision", "0.0", "1.0"),
        ("collision", "", "1.0"),
        ("collision", "2.0", ""),
        ("unnecessary_braking_event_count", "-1.0", ""),
    ),
)
def test_comparison_parser_rejects_invalid_aggregate_relationships(
    bundle_factory: Callable[[], Path],
    metric: str,
    mean: str,
    stdev: str,
) -> None:
    bundle = bundle_factory()

    def mutate(rows: list[dict[str, str]]) -> None:
        row = next(item for item in rows if item["metric"] == metric)
        row["mean"] = mean
        row["policy_seed_stdev"] = stdev

    payload = _mutated_csv(bundle / "metrics/comparison.csv", mutate)
    with pytest.raises(ValueError):
        report._parse_comparison_csv(payload)


def test_comparison_parser_rejects_duplicate_identity(
    bundle_factory: Callable[[], Path],
) -> None:
    bundle = bundle_factory()

    def duplicate(rows: list[dict[str, str]]) -> None:
        rows.insert(1, dict(rows[0]))

    payload = _mutated_csv(bundle / "metrics/comparison.csv", duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        report._parse_comparison_csv(payload)


def test_report_csv_fixtures_retain_expected_columns(
    bundle_factory: Callable[[], Path],
) -> None:
    bundle = bundle_factory()
    for relative, expected in (
        ("metrics/train_metrics.csv", TRAIN_METRICS_CSV_COLUMNS),
        ("metrics/eval_metrics.csv", EVAL_METRICS_CSV_COLUMNS),
        ("metrics/comparison.csv", COMPARISON_CSV_COLUMNS),
    ):
        with (bundle / relative).open(encoding="utf-8", newline="") as source:
            assert tuple(csv.DictReader(source).fieldnames or ()) == expected
