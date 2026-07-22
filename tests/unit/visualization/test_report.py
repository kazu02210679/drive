from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from mad_driving.visualization.report import write_markdown_report


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
