from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from mad_driving.visualization.report import write_markdown_report


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames is not None
        return reader.fieldnames, list(reader)


def _rewrite_csv_and_manifest(
    bundle: Path,
    relative: str,
    mutate: Callable[[list[dict[str, str]]], None],
) -> None:
    path = bundle / relative
    columns, rows = _read_csv_rows(path)
    mutate(rows)
    (bundle / "evaluation_manifest.json").unlink()
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    from mad_driving.evaluation.workspace import EvaluationWorkspace

    EvaluationWorkspace(destination=bundle, path=bundle).write_manifest()


def _rewrite_json_and_manifest(
    bundle: Path,
    relative: str,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    path = bundle / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    mutate(payload)
    (bundle / "evaluation_manifest.json").unlink()
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    from mad_driving.evaluation.workspace import EvaluationWorkspace

    EvaluationWorkspace(destination=bundle, path=bundle).write_manifest()


def _make_formal(bundle: Path) -> None:
    def train(rows: list[dict[str, str]]) -> None:
        for row in rows:
            row["result_label"] = ""
            if row["value"] == "":
                row["value"] = "0.0"

    def evaluated(rows: list[dict[str, str]]) -> None:
        for row in rows:
            row["result_label"] = ""
            row["is_formal"] = "True"

    _rewrite_csv_and_manifest(bundle, "metrics/train_metrics.csv", train)
    _rewrite_csv_and_manifest(bundle, "metrics/eval_metrics.csv", evaluated)
    _rewrite_csv_and_manifest(bundle, "metrics/comparison.csv", evaluated)

    def checkpoints(payload: dict[str, object]) -> None:
        selected = payload["selected_checkpoints"]
        assert isinstance(selected, list)
        selected.insert(
            0,
            {
                "checkpoint_path": "runs/b1_nominal/42.zip",
                "checkpoint_sha256": "a" * 64,
                "method_id": "b1_nominal",
                "policy_seed": 42,
                "validation_plan_sha256": "b" * 64,
            },
        )

    _rewrite_json_and_manifest(bundle, "selected_checkpoints.json", checkpoints)


@pytest.mark.parametrize("existing", [False, True])
def test_every_writer_rejects_output_inside_verified_source_bundle(
    bundle_factory: Callable[[], Path], existing: bool
) -> None:
    from mad_driving.visualization.overlay import write_episode_gif
    from mad_driving.visualization.plots import (
        write_learning_curve,
        write_safety_efficiency_plots,
    )

    bundle = bundle_factory()
    trace = next(bundle.glob("episodes/**/*_trace.jsonl"))
    frames = trace.parent / "episode_20001_frames"
    existing_path = bundle / "config_resolved.yaml"
    cases = (
        (
            write_learning_curve,
            (
                bundle / "metrics" / "train_metrics.csv",
                existing_path if existing else bundle / "new.png",
            ),
        ),
        (
            write_safety_efficiency_plots,
            (bundle / "metrics" / "eval_metrics.csv", bundle if existing else bundle / "new-plots"),
        ),
        (
            write_episode_gif,
            (trace, frames, existing_path if existing else bundle / "new.gif"),
        ),
        (
            write_markdown_report,
            (bundle, existing_path if existing else bundle / "new.md"),
        ),
    )

    for writer, arguments in cases:
        with pytest.raises(ValueError, match="source bundle|inside|containment"):
            writer(*arguments)


def test_every_writer_rejects_output_parent_symlink_containment_when_supported(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    from mad_driving.visualization.overlay import write_episode_gif
    from mad_driving.visualization.plots import (
        write_learning_curve,
        write_safety_efficiency_plots,
    )

    bundle = bundle_factory()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(bundle, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink creation unavailable: {error}")
    trace = next(bundle.glob("episodes/**/*_trace.jsonl"))
    frames = trace.parent / "episode_20001_frames"
    cases = (
        (
            write_learning_curve,
            (bundle / "metrics" / "train_metrics.csv", linked_parent / "new.png"),
        ),
        (
            write_safety_efficiency_plots,
            (bundle / "metrics" / "eval_metrics.csv", linked_parent / "plots"),
        ),
        (write_episode_gif, (trace, frames, linked_parent / "new.gif")),
        (write_markdown_report, (bundle, linked_parent / "new.md")),
    )

    for writer, arguments in cases:
        with pytest.raises(ValueError, match="symbolic|reparse|source bundle|containment"):
            writer(*arguments)


def test_report_is_deterministic_complete_and_uses_relative_verified_links(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    first = bundle_factory()
    output = tmp_path / "reports" / "comparison_report.md"

    write_markdown_report(first, output)
    report = output.read_text(encoding="utf-8")

    assert report.startswith("# Phase 6 Evaluation Report\n")
    assert "SMOKE - NOT A RESEARCH RESULT" in report
    assert "pipeline validation, not a formal scientific conclusion" in report
    assert "[Artifact manifest](../verified-bundle/evaluation_manifest.json)" in report
    assert "[Evaluation plan](../verified-bundle/evaluation_plan.yaml)" in report
    assert "[Selected checkpoints](../verified-bundle/selected_checkpoints.json)" in report
    assert "a" * 64 in report
    assert "## Exact evaluation matrix" in report
    assert "level1_lead_brake" in report and "20001" in report
    for track in ("Decision track", "System track", "Ablation track"):
        assert f"## {track}" in report
    for group in ("Safety", "Efficiency", "Comfort", "Multi-Agent", "Latency"):
        assert f"### {group}" in report
    for plot in (
        "learning_curve.png",
        "collision_rate.png",
        "success_route_completion.png",
        "unnecessary_braking.png",
        "comfort.png",
        "agent_disagreement.png",
    ):
        assert plot in report
    assert "proposed_42_level1_lead_brake_20001.gif" in report
    assert "N/A means the canonical source cell was empty" in report
    assert "Failures and exclusions" in report
    assert "None recorded in the verified evaluation manifest." in report
    assert output.read_bytes().endswith(b"\n")
    assert b"\r\n" not in output.read_bytes()


def test_report_repeats_bytes_and_uses_canonical_case_order(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()
    outputs = (tmp_path / "repeat-a" / "report.md", tmp_path / "repeat-b" / "report.md")

    for output in outputs:
        write_markdown_report(bundle, output)

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    report = outputs[0].read_text(encoding="utf-8")
    safety = report.split("### Safety", 1)[1].split("### Efficiency", 1)[0]
    positions = [
        safety.index(f"| {case_id} |")
        for case_id in (
            "level0_nominal",
            "level1_lead_brake",
            "level2_lead_brake",
            "level2_cut_in",
            "level3_occluded_crossing",
        )
    ]
    assert positions == sorted(positions)


def test_report_rejects_tampered_bundle_and_never_overwrites(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()
    checkpoint = bundle / "selected_checkpoints.json"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    output = tmp_path / "report.md"
    with pytest.raises(ValueError, match="manifest|verification|integrity"):
        write_markdown_report(bundle, output)
    assert not output.exists()

    clean = bundle_factory()
    output.write_text("owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_markdown_report(clean, output)
    assert output.read_text(encoding="utf-8") == "owned\n"


def test_report_rejects_noncanonical_csv_and_nonfinite_cells(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()
    evaluation = bundle / "metrics" / "eval_metrics.csv"
    text = evaluation.read_text(encoding="utf-8").replace(",2.4,", ",NaN,", 1)
    (bundle / "evaluation_manifest.json").unlink()
    evaluation.write_text(text, encoding="utf-8")
    from mad_driving.evaluation.workspace import EvaluationWorkspace

    EvaluationWorkspace(destination=bundle, path=bundle).write_manifest()
    with pytest.raises(ValueError, match="finite|NaN|numeric"):
        write_markdown_report(bundle, tmp_path / "nonfinite.md")


def test_report_does_not_invent_failures_from_untrusted_files(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()
    (bundle / "failures.txt").write_text("invented failure", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|inventory|undeclared"):
        write_markdown_report(bundle, tmp_path / "failure.md")


def test_offline_modules_import_without_simulator_training_or_tensorboard() -> None:
    script = r"""
import importlib.abc
import sys

FORBIDDEN = ("metadrive", "stable_baselines3", "tensorboard")

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in FORBIDDEN):
            raise AssertionError(f"forbidden offline import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import mad_driving.evaluation.compare
import mad_driving.visualization
import mad_driving.visualization.overlay
import mad_driving.visualization.plots
import mad_driving.visualization.report
import matplotlib
assert matplotlib.get_backend().lower() == "agg"
assert not any(
    module == name or module.startswith(name + ".")
    for module in sys.modules
    for name in FORBIDDEN
)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_report_rejects_malformed_checkpoint_json_after_valid_integrity(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()
    checkpoint = bundle / "selected_checkpoints.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["selected_checkpoints"][0]["checkpoint_sha256"] = "not-a-hash"
    (bundle / "evaluation_manifest.json").unlink()
    checkpoint.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    from mad_driving.evaluation.workspace import EvaluationWorkspace

    EvaluationWorkspace(destination=bundle, path=bundle).write_manifest()
    with pytest.raises(ValueError, match="checkpoint|SHA-256|hash"):
        write_markdown_report(bundle, tmp_path / "malformed.md")


def test_plots_reject_duplicate_canonical_train_and_eval_row_identities(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    from mad_driving.visualization.plots import (
        write_learning_curve,
        write_safety_efficiency_plots,
    )

    train_bundle = bundle_factory()
    _rewrite_csv_and_manifest(
        train_bundle,
        "metrics/train_metrics.csv",
        lambda rows: rows.append(dict(rows[-1])),
    )
    with pytest.raises(ValueError, match="duplicate.*train|training.*duplicate|identity"):
        write_learning_curve(
            train_bundle / "metrics" / "train_metrics.csv",
            tmp_path / "duplicate-train.png",
        )

    eval_bundle = bundle_factory()
    _rewrite_csv_and_manifest(
        eval_bundle,
        "metrics/eval_metrics.csv",
        lambda rows: rows.append(dict(rows[0])),
    )
    with pytest.raises(ValueError, match="duplicate.*eval|episode.*duplicate|identity"):
        write_safety_efficiency_plots(
            eval_bundle / "metrics" / "eval_metrics.csv",
            tmp_path / "duplicate-eval",
        )


@pytest.mark.parametrize(
    ("relative", "column", "value", "message"),
    (
        ("metrics/eval_metrics.csv", "record_schema_version", "", "required|integer"),
        ("metrics/eval_metrics.csv", "critic_enabled", "1", "boolean|critic_enabled"),
        ("metrics/comparison.csv", "physical_episode_count", "1.5", "integer|count"),
        ("metrics/comparison.csv", "policy_replicate_count", "-1", "positive|count"),
    ),
)
def test_report_rejects_noncanonical_required_csv_domains(
    bundle_factory: Callable[[], Path],
    tmp_path: Path,
    relative: str,
    column: str,
    value: str,
    message: str,
) -> None:
    bundle = bundle_factory()

    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0][column] = value

    _rewrite_csv_and_manifest(bundle, relative, mutate)
    with pytest.raises(ValueError, match=message):
        write_markdown_report(bundle, tmp_path / f"invalid-{column}.md")


def test_report_rejects_mixed_smoke_formality_even_with_valid_labels(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()

    def mutate(rows: list[dict[str, str]]) -> None:
        rows[0]["result_label"] = ""
        rows[0]["is_formal"] = "True"

    _rewrite_csv_and_manifest(bundle, "metrics/comparison.csv", mutate)
    with pytest.raises(ValueError, match="formal|smoke|result_label|inconsistent"):
        write_markdown_report(bundle, tmp_path / "mixed.md")


def test_report_rejects_one_missing_expected_comparison_matrix_row(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()
    _rewrite_csv_and_manifest(
        bundle,
        "metrics/comparison.csv",
        lambda rows: rows.pop(),
    )

    with pytest.raises(ValueError, match="comparison.*matrix|missing.*comparison|required.*row"):
        write_markdown_report(bundle, tmp_path / "missing-comparison.md")


def test_smoke_report_explicitly_allows_unselected_ppo_checkpoints(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()

    def clear(payload: dict[str, object]) -> None:
        payload["selected_checkpoints"] = []

    _rewrite_json_and_manifest(bundle, "selected_checkpoints.json", clear)
    output = tmp_path / "smoke-unselected.md"
    write_markdown_report(bundle, output)
    assert "SMOKE - NOT A RESEARCH RESULT" in output.read_text(encoding="utf-8")


def test_formal_report_rejects_eval_checkpoint_not_bound_by_selection(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()
    _make_formal(bundle)

    def mismatch(rows: list[dict[str, str]]) -> None:
        next(row for row in rows if row["method_id"] == "proposed")["checkpoint_sha256"] = "c" * 64

    _rewrite_csv_and_manifest(bundle, "metrics/eval_metrics.csv", mismatch)
    message = "formal.*checkpoint|selected.*checkpoint|checkpoint.*hash"
    with pytest.raises(ValueError, match=message):
        write_markdown_report(bundle, tmp_path / "formal-mismatch.md")


def test_report_rejects_inconsistent_selected_validation_plan_provenance(
    bundle_factory: Callable[[], Path], tmp_path: Path
) -> None:
    bundle = bundle_factory()
    _make_formal(bundle)

    def mismatch(payload: dict[str, object]) -> None:
        selected = payload["selected_checkpoints"]
        assert isinstance(selected, list)
        assert isinstance(selected[0], dict)
        selected[0]["validation_plan_sha256"] = "c" * 64

    _rewrite_json_and_manifest(bundle, "selected_checkpoints.json", mismatch)
    with pytest.raises(ValueError, match="validation.plan|provenance"):
        write_markdown_report(bundle, tmp_path / "plan-mismatch.md")
