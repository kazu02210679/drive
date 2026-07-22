"""Evaluate the strict Phase 6 method matrix into a fresh artifact bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from mad_driving.config.loader import load_config
from mad_driving.evaluation.models import EvaluationPlanConfig
from mad_driving.evaluation.selection import validate_ppo_checkpoint_archive
from mad_driving.evaluation.serialization import load_evaluation_plan
from mad_driving.visualization import METHOD_ORDER, SMOKE_RESULT_LABEL


def run_evaluation_bundle(
    *,
    evaluation_config: EvaluationPlanConfig,
    plan_path: Path,
    app_config_path: Path,
    method_overlays: tuple[Path, ...],
    checkpoint_paths: Mapping[tuple[str, int], Path],
    destination: Path,
    smoke: bool,
) -> Path:
    """Injected Task 10 orchestration seam."""

    del (
        evaluation_config,
        plan_path,
        app_config_path,
        method_overlays,
        checkpoint_paths,
        destination,
        smoke,
    )
    raise RuntimeError("evaluation bundle orchestration is not installed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Strict evaluation plan YAML")
    parser.add_argument("--output", required=True, help="Fresh evaluation bundle destination")
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Ordered method/config overlay appended after plan overlays; repeat as needed",
    )
    parser.add_argument("--smoke", action="store_true", help="Require the explicit smoke plan")
    return parser


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path)))


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path.resolve(strict=True)


def _require_directory(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path.resolve(strict=True)


def _require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")


def _validate_mode(config: EvaluationPlanConfig, *, smoke: bool) -> str:
    expects_smoke = config.plan_kind == "phase6_smoke"
    if smoke is not expects_smoke:
        raise ValueError("--smoke must be used exactly for a phase6_smoke plan")
    if expects_smoke:
        if config.is_formal is not False:
            raise ValueError("smoke plan must explicitly set is_formal: false")
        if config.result_label != SMOKE_RESULT_LABEL:
            raise ValueError(f"smoke plan result_label must be {SMOKE_RESULT_LABEL!r}")
        if config.max_episode_steps is None:
            raise ValueError("smoke plan must set a finite positive max_episode_steps")
        return SMOKE_RESULT_LABEL
    if config.is_formal is False:
        raise ValueError("formal plan cannot set is_formal: false")
    return config.result_label or "FORMAL RESEARCH RESULT"


def _validate_method_overlays(app_config_path: Path, overlays: tuple[Path, ...]) -> None:
    if len(overlays) != len(METHOD_ORDER):
        raise ValueError("evaluation plan must declare the exact ordered method overlays")
    for expected_method, overlay in zip(METHOD_ORDER, overlays, strict=True):
        resolved = load_config(app_config_path, overlay)
        if resolved.method.id != expected_method:
            raise ValueError(
                f"method overlay must identify {expected_method}: {overlay}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with validation before orchestration ownership."""

    args = _parser().parse_args(argv)
    try:
        plan_path = _require_file(_absolute(args.plan), "Evaluation plan")
        destination = _absolute(args.output)
        _require_absent(destination)
        config = load_evaluation_plan(plan_path)
        result_label = _validate_mode(config, smoke=args.smoke)
        app_config_path = _require_file(_absolute(config.app_config_path), "App configuration")
        plan_overlays = tuple(
            _require_file(_absolute(path), "Method overlay") for path in config.method_overlays
        )
        _validate_method_overlays(app_config_path, plan_overlays)
        cli_overlays = tuple(
            _require_file(_absolute(path), "Method overlay") for path in args.overlay
        )
        checkpoint_paths: dict[tuple[str, int], Path] = {}
        for binding in config.ppo_run_bindings:
            training_run = _require_directory(
                _absolute(binding.training_run_dir), "Completed training run"
            )
            if binding.checkpoint_path is None:
                raise ValueError(
                    "PPO checkpoint binding is missing for "
                    f"{binding.method_id}/{binding.policy_seed}"
                )
            checkpoint = _require_file(
                _absolute(binding.checkpoint_path), "PPO checkpoint"
            )
            try:
                checkpoint.relative_to(training_run)
            except ValueError as error:
                raise ValueError(
                    f"PPO checkpoint is outside its bound training run: {checkpoint}"
                ) from error
            validate_ppo_checkpoint_archive(checkpoint)
            checkpoint_paths[(binding.method_id, binding.policy_seed)] = checkpoint
        published = run_evaluation_bundle(
            evaluation_config=config,
            plan_path=plan_path,
            app_config_path=app_config_path,
            method_overlays=plan_overlays + cli_overlays,
            checkpoint_paths=checkpoint_paths,
            destination=destination,
            smoke=args.smoke,
        )
        output = {"output": str(published), "result_label": result_label}
    except Exception as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
