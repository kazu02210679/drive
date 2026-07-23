"""Evaluate the strict Phase 6 method matrix into a fresh artifact bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from mad_driving.cli._common import concise_operational_error, validate_absent_destination
from mad_driving.config.loader import load_config
from mad_driving.config.models import AppConfig
from mad_driving.evaluation.models import (
    EvaluationRunSpec,
    Phase6PublicationPlan,
)
from mad_driving.evaluation.plans import build_formal_plan, build_smoke_plan
from mad_driving.evaluation.selection import (
    CheckpointCandidate,
    discover_checkpoint_candidates,
    validate_ppo_checkpoint_archive,
)
from mad_driving.evaluation.serialization import load_phase6_publication_plan
from mad_driving.evaluation.training_metrics import extract_training_metrics
from mad_driving.methods import MethodProfileSnapshot
from mad_driving.visualization import METHOD_ORDER, SMOKE_RESULT_LABEL

if TYPE_CHECKING:
    from mad_driving.evaluation.bundle import (
        CheckpointReader,
        EvaluationEnvironmentFactory,
        EvaluationPolicyFactory,
        RgbFrameProvider,
        TrainingEventReader,
    )
    from mad_driving.evaluation.selection import CheckpointScore


def run_evaluation_bundle(
    *,
    evaluation_config: Phase6PublicationPlan,
    plan_path: Path,
    run_plan: tuple[EvaluationRunSpec, ...],
    method_configs: tuple[AppConfig, ...],
    method_profiles: tuple[MethodProfileSnapshot, ...],
    cli_overlays: tuple[Path, ...],
    authenticated_checkpoints: tuple[CheckpointCandidate, ...],
    destination: Path,
    smoke: bool,
    environment_factory: EvaluationEnvironmentFactory | None = None,
    policy_factory: EvaluationPolicyFactory | None = None,
    frame_provider: RgbFrameProvider | None = None,
    selection_scores: tuple[CheckpointScore, ...] | None = None,
    checkpoint_reader: CheckpointReader | None = None,
    event_reader: TrainingEventReader | None = None,
) -> Path:
    """Delegate to the Task 10 orchestrator with Task 11 factories injected later."""

    from mad_driving.evaluation.bundle import run_evaluation_bundle as implementation

    if (
        environment_factory is None
        or policy_factory is None
        or frame_provider is None
        or selection_scores is None
    ):
        raise RuntimeError("real evaluation factories are not installed until Task 11")
    return implementation(
        evaluation_config=evaluation_config,
        plan_path=plan_path,
        run_plan=run_plan,
        method_configs=method_configs,
        method_profiles=method_profiles,
        cli_overlays=cli_overlays,
        authenticated_checkpoints=authenticated_checkpoints,
        destination=destination,
        smoke=smoke,
        environment_factory=environment_factory,
        policy_factory=policy_factory,
        frame_provider=frame_provider,
        selection_scores=selection_scores,
        checkpoint_reader=checkpoint_reader or validate_ppo_checkpoint_archive,
        event_reader=event_reader or extract_training_metrics,
    )


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


def _validate_mode(config: Phase6PublicationPlan, *, smoke: bool) -> str:
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
    return config.result_label


def _resolved_method_configs(
    app_config_path: Path,
    method_overlays: tuple[Path, ...],
    cli_overlays: tuple[Path, ...],
) -> tuple[AppConfig, ...]:
    if len(method_overlays) != len(METHOD_ORDER):
        raise ValueError("evaluation plan must declare the exact ordered method overlays")
    resolved_configs: list[AppConfig] = []
    for expected_method, method_overlay in zip(METHOD_ORDER, method_overlays, strict=True):
        resolved = load_config(app_config_path, method_overlay, *cli_overlays)
        if resolved.method.id != expected_method:
            raise ValueError(f"method overlay must resolve to {expected_method}: {method_overlay}")
        resolved_configs.append(resolved)
    return tuple(resolved_configs)


def _authenticate_checkpoints(
    config: Phase6PublicationPlan,
) -> tuple[CheckpointCandidate, ...]:
    authenticated: list[CheckpointCandidate] = []
    for binding in config.ppo_run_bindings:
        training_run = _require_directory(
            _absolute(binding.training_run_dir), "Completed training run"
        )
        checkpoint = _require_file(_absolute(binding.checkpoint_path), "PPO checkpoint")
        try:
            checkpoint.relative_to(training_run)
        except ValueError as error:
            raise ValueError(
                f"PPO checkpoint is outside its bound training run: {checkpoint}"
            ) from error
        matches = tuple(
            candidate
            for candidate in discover_checkpoint_candidates(training_run)
            if candidate.path.resolve(strict=True) == checkpoint
            and candidate.method_id == binding.method_id
            and candidate.policy_seed == binding.policy_seed
        )
        if len(matches) != 1:
            raise ValueError(
                "PPO checkpoint must match exactly one authenticated training-run candidate: "
                f"{binding.method_id}/{binding.policy_seed}"
            )
        authenticated.append(matches[0])
    return tuple(authenticated)


def _build_run_plan(
    config: Phase6PublicationPlan,
    checkpoints: tuple[CheckpointCandidate, ...],
) -> tuple[EvaluationRunSpec, ...]:
    checkpoint_paths = {
        (candidate.method_id, candidate.policy_seed): str(candidate.path)
        for candidate in checkpoints
    }
    if config.plan_kind == "phase6_smoke":
        return build_smoke_plan(config, checkpoint_paths)
    return build_formal_plan(config, checkpoint_paths)


def _validate_cli_overlays(paths: tuple[Path, ...]) -> None:
    if len(paths) != len(set(paths)):
        raise ValueError("CLI overlays must not contain duplicate paths")


def _method_profiles() -> tuple[MethodProfileSnapshot, ...]:
    return tuple(MethodProfileSnapshot.from_method_id(method_id) for method_id in METHOD_ORDER)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with validation before orchestration ownership."""

    args = _parser().parse_args(argv)
    try:
        plan_path = _require_file(_absolute(args.plan), "Evaluation plan")
        destination = validate_absent_destination(_absolute(args.output))
        config = load_phase6_publication_plan(plan_path)
        result_label = _validate_mode(config, smoke=args.smoke)
        app_config_path = _require_file(_absolute(config.app_config_path), "App configuration")
        plan_overlays = tuple(
            _require_file(_absolute(path), "Method overlay") for path in config.method_overlays
        )
        cli_overlays = tuple(
            _require_file(_absolute(path), "Method overlay") for path in args.overlay
        )
        _validate_cli_overlays(cli_overlays)
        method_configs = _resolved_method_configs(app_config_path, plan_overlays, cli_overlays)
        authenticated_checkpoints = _authenticate_checkpoints(config)
        validate_absent_destination(
            destination,
            source_roots=tuple(
                _absolute(binding.training_run_dir) for binding in config.ppo_run_bindings
            ),
        )
        run_plan = _build_run_plan(config, authenticated_checkpoints)
        published = run_evaluation_bundle(
            evaluation_config=config,
            plan_path=plan_path,
            run_plan=run_plan,
            method_configs=method_configs,
            method_profiles=_method_profiles(),
            cli_overlays=cli_overlays,
            authenticated_checkpoints=authenticated_checkpoints,
            destination=destination,
            smoke=args.smoke,
        )
        output = {"output": str(published), "result_label": result_label}
    except Exception as exc:
        print(f"evaluation failed: {concise_operational_error(exc)}", file=sys.stderr)
        return 2
    print(json.dumps(output, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
