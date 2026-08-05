"""Generate an offline comparison bundle from a verified evaluation bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from mad_driving.cli._common import concise_operational_error
from mad_driving.visualization import _reject_output_in_source_bundle, _verify_bundle


def run_comparison_bundle(*, evaluation: Path, destination: Path) -> Path:
    """Regenerate comparison plots and Markdown from one verified source bundle."""

    from mad_driving.evaluation.bundle import _cleanup_workspace
    from mad_driving.evaluation.workspace import EvaluationWorkspace
    from mad_driving.visualization import (
        write_learning_curve,
        write_safety_efficiency_plots,
    )
    from mad_driving.visualization.report import write_markdown_report

    workspace: EvaluationWorkspace | None = None
    try:
        workspace = EvaluationWorkspace.stage(destination)
        write_learning_curve(
            evaluation / "metrics" / "train_metrics.csv",
            workspace.path / "plots" / "learning_curve.png",
        )
        write_safety_efficiency_plots(
            evaluation / "metrics" / "eval_metrics.csv",
            workspace.path / "plots",
        )
        write_markdown_report(
            evaluation,
            workspace.path / "comparison_report.md",
        )
        workspace.write_manifest()
        return workspace.publish()
    except BaseException:
        _cleanup_workspace(workspace)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, help="Verified evaluation bundle")
    parser.add_argument(
        "--output",
        help="Fresh output directory; defaults to <evaluation>-comparison beside the source",
    )
    return parser


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path)))


def _destination(evaluation: Path, supplied: str | None) -> Path:
    if supplied is not None:
        return _absolute(supplied)
    return evaluation.with_name(f"{evaluation.name}-comparison")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point that never mutates the verified source bundle."""

    args = _parser().parse_args(argv)
    try:
        evaluation = _absolute(args.evaluation)
        destination = _destination(evaluation, args.output)
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        bundle = _verify_bundle(evaluation)
        _reject_output_in_source_bundle(bundle.root, destination)
        published = run_comparison_bundle(
            evaluation=bundle.root,
            destination=destination,
        )
        output = {"output": str(published)}
    except Exception as exc:
        print(f"comparison failed: {concise_operational_error(exc)}", file=sys.stderr)
        return 2
    print(json.dumps(output, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
