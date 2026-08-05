"""Two-stage orchestration for one complete immutable Phase 6 bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Protocol

import yaml

from mad_driving.config.models import AppConfig
from mad_driving.evaluation.compare import (
    build_comparison_rows,
    validate_matched_episodes,
    write_comparison_csv,
    write_eval_metrics_csv,
)
from mad_driving.evaluation.metrics import EpisodeMetricRecord, reduce_episode
from mad_driving.evaluation.models import EvaluationRunSpec, Phase6PublicationPlan
from mad_driving.evaluation.paths import validate_absent_destination
from mad_driving.evaluation.plans import build_formal_plan, build_smoke_plan
from mad_driving.evaluation.policies import EvaluationPolicy
from mad_driving.evaluation.runner import EvaluationEnvironment, run_evaluation_episode
from mad_driving.evaluation.selection import (
    CheckpointCandidate,
    CheckpointScore,
    resolved_config_sha256,
    validate_ppo_checkpoint_archive,
    write_selection_artifacts,
    write_unselected_smoke_checkpoint_artifacts,
)
from mad_driving.evaluation.serialization import (
    load_phase6_publication_plan,
    write_jsonl_strict,
)
from mad_driving.evaluation.training_metrics import (
    TensorBoardEventSource,
    TrainingMetricPoint,
    _event_files,
    extract_training_metrics_from_event_sources,
    write_training_metrics_csv,
)
from mad_driving.evaluation.workspace import EvaluationWorkspace
from mad_driving.methods import MethodProfileSnapshot
from mad_driving.visualization import (
    METHOD_ORDER,
    _read_stable_regular_file,
    _verify_bundle,
    write_episode_gif,
    write_learning_curve,
    write_safety_efficiency_plots,
)
from mad_driving.visualization.report import write_staged_markdown_report


class EvaluationEnvironmentFactory(Protocol):
    def __call__(self, spec: EvaluationRunSpec, config: AppConfig) -> EvaluationEnvironment: ...


class EvaluationPolicyFactory(Protocol):
    def __call__(
        self,
        spec: EvaluationRunSpec,
        config: AppConfig,
        candidate: CheckpointCandidate | None,
    ) -> EvaluationPolicy: ...


class TrainingEventReader(Protocol):
    def __call__(
        self, sources: Sequence[TensorBoardEventSource], *, smoke: bool
    ) -> tuple[TrainingMetricPoint, ...]: ...


class RgbFrameProvider(Protocol):
    def __call__(self, spec: EvaluationRunSpec, record_count: int) -> tuple[bytes, ...]: ...


CheckpointReader = Callable[[Path], str]


def _profile_payload(profile: MethodProfileSnapshot) -> dict[str, object]:
    return {
        "critic_enabled": profile.critic_enabled,
        "method_id": profile.method_id,
        "policy_kind": profile.policy_kind,
        "shield_mode": profile.shield_mode,
        "specialist_ids": list(profile.specialist_ids),
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _write_json(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes(path, encoded)


def _episode_key_text(spec: EvaluationRunSpec) -> str:
    policy_seed = "rule" if spec.policy_seed is None else str(spec.policy_seed)
    return "_".join(
        (
            spec.method_id,
            spec.track,
            policy_seed,
            spec.scenario_cell_id,
            str(spec.test_seed),
        )
    )


def _episode_directory(root: Path, spec: EvaluationRunSpec) -> Path:
    policy_seed = "rule" if spec.policy_seed is None else str(spec.policy_seed)
    return root.joinpath(
        "episodes",
        spec.method_id,
        spec.track,
        policy_seed,
        spec.scenario_cell_id,
    )


def _remove_owned_tree(path: Path, owner: Path) -> None:
    if not path.exists():
        return
    owner_resolved = owner.resolve(strict=True)
    target = path.resolve(strict=True)
    try:
        relative = target.relative_to(owner_resolved)
    except ValueError as error:
        raise RuntimeError(f"refusing to clean an unowned private path: {target}") from error
    if not relative.parts:
        raise RuntimeError("refusing to clean the private workspace root as a child")
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"private cleanup target is not an owned regular directory: {path}")
    shutil.rmtree(path)


def _cleanup_workspace(workspace: EvaluationWorkspace | None) -> None:
    if workspace is None or not workspace.path.exists():
        return
    target = workspace.path.resolve(strict=True)
    parent = workspace.destination.parent.resolve(strict=True)
    try:
        relative = target.relative_to(parent)
    except ValueError as error:
        raise RuntimeError(f"refusing to clean an unowned staging path: {target}") from error
    prefix = f".{workspace.destination.name}.staging-"
    if len(relative.parts) != 1 or not target.name.startswith(prefix):
        raise RuntimeError(f"refusing to clean an unproven staging path: {target}")
    metadata = workspace.path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"staging cleanup target is not a regular directory: {target}")
    shutil.rmtree(workspace.path)


def _validated_inputs(
    *,
    evaluation_config: Phase6PublicationPlan,
    plan_path: Path,
    run_plan: tuple[EvaluationRunSpec, ...],
    method_configs: tuple[AppConfig, ...],
    method_profiles: tuple[MethodProfileSnapshot, ...],
    authenticated_checkpoints: tuple[CheckpointCandidate, ...],
    checkpoint_reader: CheckpointReader,
    selection_scores: tuple[CheckpointScore, ...] | None,
    smoke: bool,
) -> tuple[dict[str, AppConfig], dict[tuple[str, int], CheckpointCandidate]]:
    loaded_plan = load_phase6_publication_plan(plan_path)
    if loaded_plan != evaluation_config:
        raise ValueError("evaluation plan path does not match the validated publication plan")
    if smoke is not (evaluation_config.plan_kind == "phase6_smoke"):
        raise ValueError("smoke mode does not match the validated publication plan")
    expected_profiles = tuple(
        MethodProfileSnapshot.from_method_id(method_id) for method_id in METHOD_ORDER
    )
    if method_profiles != expected_profiles:
        raise ValueError("method profiles do not match the exact Phase 6 method order")
    if tuple(config.method.id for config in method_configs) != METHOD_ORDER:
        raise ValueError("resolved method configs do not match the exact Phase 6 method order")
    configs: dict[str, AppConfig] = {config.method.id: config for config in method_configs}

    candidates: dict[tuple[str, int], CheckpointCandidate] = {}
    for candidate in authenticated_checkpoints:
        key = (candidate.method_id, candidate.policy_seed)
        if key in candidates:
            raise ValueError("authenticated checkpoints contain a duplicate method/policy seed")
        base_config = configs[candidate.method_id]
        expected_config = base_config.model_copy(
            update={
                "training": base_config.training.model_copy(update={"seed": candidate.policy_seed})
            }
        )
        expected_config_sha256 = resolved_config_sha256(expected_config.model_dump(mode="json"))
        if candidate.resolved_config_sha256 != expected_config_sha256:
            raise ValueError(
                "authenticated checkpoint resolved config does not match evaluation config"
            )
        if checkpoint_reader(candidate.path) != candidate.sha256:
            raise ValueError("authenticated checkpoint bytes do not match their SHA-256 binding")
        candidates[key] = candidate
    expected_bindings: dict[tuple[str, int], Path] = {
        (binding.method_id, binding.policy_seed): Path(os.path.abspath(binding.checkpoint_path))
        for binding in evaluation_config.ppo_run_bindings
    }
    if set(candidates) != set(expected_bindings) or any(
        Path(os.path.abspath(candidate.path)) != expected_bindings[key]
        for key, candidate in candidates.items()
    ):
        raise ValueError("authenticated checkpoints do not exactly match plan bindings")

    checkpoint_paths = {key: str(candidate.path) for key, candidate in candidates.items()}
    expected_plan = (
        build_smoke_plan(evaluation_config, checkpoint_paths)
        if smoke
        else build_formal_plan(evaluation_config, checkpoint_paths)
    )
    if run_plan != expected_plan:
        raise ValueError("run plan does not match the complete validated Phase 6 matrix")

    if selection_scores is None:
        if not smoke:
            raise ValueError("formal evaluation requires authenticated checkpoint-selection scores")
    else:
        score_candidates = tuple(score.candidate for score in selection_scores)
        if set(score_candidates) != set(authenticated_checkpoints):
            raise ValueError("selection scores do not exactly bind authenticated checkpoints")
    return configs, candidates


def _write_source_provenance(
    root: Path,
    *,
    evaluation_config: Phase6PublicationPlan,
    plan_path: Path,
    method_configs: tuple[AppConfig, ...],
    method_profiles: tuple[MethodProfileSnapshot, ...],
    cli_overlays: tuple[Path, ...],
) -> None:
    plan_bytes = _read_stable_regular_file(plan_path, label="evaluation plan")
    if load_phase6_publication_plan(plan_path) != evaluation_config:
        raise ValueError("evaluation plan changed while preparing the source snapshot")
    _write_bytes(root / "evaluation_plan.yaml", plan_bytes)
    config_payload = {
        "cli_overlays": [str(path) for path in cli_overlays],
        "evaluation_id": evaluation_config.evaluation_id,
        "methods": [
            {
                "config": config.model_dump(mode="json"),
                "profile": _profile_payload(profile),
            }
            for config, profile in zip(method_configs, method_profiles, strict=True)
        ],
        "research_contract_version": 7,
    }
    encoded = yaml.safe_dump(
        config_payload,
        allow_unicode=True,
        sort_keys=True,
    ).encode("utf-8")
    _write_bytes(root / "config_resolved.yaml", encoded)


def _copy_tensorboard_sources(
    root: Path, evaluation_config: Phase6PublicationPlan
) -> tuple[TensorBoardEventSource, ...]:
    copied_sources: list[TensorBoardEventSource] = []
    bindings = sorted(
        evaluation_config.ppo_run_bindings,
        key=lambda binding: (METHOD_ORDER.index(binding.method_id), binding.policy_seed),
    )
    for binding in bindings:
        run_dir = Path(binding.training_run_dir)
        tensorboard_dir = run_dir / "tensorboard"
        for source in _event_files(tensorboard_dir):
            relative = source.relative_to(tensorboard_dir)
            destination = root.joinpath(
                "sources",
                binding.method_id,
                str(binding.policy_seed),
                "tensorboard",
                relative,
            )
            payload = _read_stable_regular_file(source, label=source.as_posix())
            _write_bytes(destination, payload)
            copied_sources.append(
                TensorBoardEventSource(
                    run_id=run_dir.name,
                    method_id=binding.method_id,
                    policy_seed=binding.policy_seed,
                    event_relative_path=relative.as_posix(),
                    payload=payload,
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
    return tuple(copied_sources)


def _validate_training_points(
    points: tuple[TrainingMetricPoint, ...],
    evaluation_config: Phase6PublicationPlan,
) -> None:
    expected = {
        (binding.method_id, binding.policy_seed): Path(binding.training_run_dir).name
        for binding in evaluation_config.ppo_run_bindings
    }
    observed: dict[tuple[str, int], str] = {}
    for point in points:
        key = (point.method_id, point.policy_seed)
        previous = observed.setdefault(key, point.run_id)
        if previous != point.run_id:
            raise ValueError("training metrics disagree on run provenance")
    if observed != expected:
        raise ValueError("training metrics do not exactly match PPO run bindings")


def _write_online_outputs(
    root: Path,
    *,
    evaluation_config: Phase6PublicationPlan,
    run_plan: tuple[EvaluationRunSpec, ...],
    configs: dict[str, AppConfig],
    candidates: dict[tuple[str, int], CheckpointCandidate],
    environment_factory: EvaluationEnvironmentFactory,
    policy_factory: EvaluationPolicyFactory,
    frame_provider: RgbFrameProvider,
) -> tuple[EpisodeMetricRecord, ...]:
    online_root = root / ".online"
    online_root.mkdir()
    capture_keys = set(evaluation_config.capture_episode_keys)
    records: list[EpisodeMetricRecord] = []
    for index, spec in enumerate(run_plan):
        config = configs[spec.method_id]
        candidate = (
            None if spec.policy_seed is None else candidates[(spec.method_id, spec.policy_seed)]
        )
        private_episode = online_root / f"{index:04d}-{_episode_key_text(spec)}"
        environment = environment_factory(spec, config)
        try:
            policy = policy_factory(spec, config, candidate)
        except BaseException as error:
            try:
                environment.close()
            except BaseException as close_error:
                error.add_note(f"environment cleanup also failed: {close_error!r}")
            raise
        result = run_evaluation_episode(
            spec,
            environment=environment,
            policy=policy,
            config=config,
            destination=private_episode,
            checkpoint_sha256=None if candidate is None else candidate.sha256,
        )
        episode_dir = _episode_directory(root, spec)
        trace = episode_dir / f"episode_{spec.test_seed}_trace.jsonl"
        summary = episode_dir / f"episode_{spec.test_seed}_summary.json"
        write_jsonl_strict(trace, (record.to_dict() for record in result.step_records))
        _write_json(summary, result.episode_record.to_dict())

        capture = _episode_key_text(spec) in capture_keys
        if capture:
            frames = frame_provider(spec, len(result.step_records))
            if len(frames) != len(result.step_records):
                raise ValueError("captured RGB frame count does not match persisted steps")
            for record, payload in zip(result.step_records, frames, strict=True):
                if record.frame_path is None:
                    raise ValueError("captured episode step is missing its canonical frame path")
                relative = PurePosixPath(record.frame_path)
                expected = episode_dir.joinpath(
                    f"episode_{spec.test_seed}_frames", f"{record.step_index:06d}.png"
                )
                if root / relative != expected:
                    raise ValueError("captured frame path does not match its episode identity")
                _write_bytes(expected, payload)
        elif any(record.frame_path is not None for record in result.step_records):
            raise ValueError("uncaptured episode unexpectedly declares persisted frames")

        metrics = reduce_episode(
            result.step_records,
            result.step_records[0].decision_interval_s,
            near_miss_ttc_s=config.reward.near_miss_ttc_s,
        )
        records.append(EpisodeMetricRecord(result.episode_record, metrics))
        _remove_owned_tree(private_episode, root)
    _remove_owned_tree(online_root, root)
    planned_capture_keys = {
        key for spec in run_plan if (key := _episode_key_text(spec)) in capture_keys
    }
    if capture_keys != planned_capture_keys:
        raise ValueError("capture_episode_keys contains an episode outside the run plan")
    return tuple(records)


def _copy_verified_source(source_root: Path, final_root: Path) -> None:
    source = _verify_bundle(source_root)
    for relative in source.artifacts:
        _write_bytes(
            final_root / PurePosixPath(relative),
            source.read_bytes(source_root / relative),
        )


def _render_final_outputs(
    source_root: Path,
    final_root: Path,
    evaluation_config: Phase6PublicationPlan,
) -> None:
    write_learning_curve(
        source_root / "metrics" / "train_metrics.csv",
        final_root / "plots" / "learning_curve.png",
    )
    write_safety_efficiency_plots(
        source_root / "metrics" / "eval_metrics.csv",
        final_root / "plots",
    )
    for episode_key in evaluation_config.capture_episode_keys:
        matching = tuple(source_root.glob("episodes/**/*_trace.jsonl"))
        trace = next(
            (
                path
                for path in matching
                if episode_key == _episode_key_from_trace(source_root, path)
            ),
            None,
        )
        if trace is None:
            raise ValueError(f"captured episode trace is missing: {episode_key}")
        frames = trace.with_name(trace.name.replace("_trace.jsonl", "_frames"))
        parts = trace.relative_to(source_root).parts
        method_id, _track, policy_seed, case_id = parts[1:5]
        episode_seed = trace.name.removeprefix("episode_").removesuffix("_trace.jsonl")
        filename = f"{method_id}_{policy_seed}_{case_id}_{episode_seed}.gif"
        output = final_root / "renders" / filename
        write_episode_gif(trace, frames, output)
    write_staged_markdown_report(
        source_bundle_dir=source_root,
        staged_bundle_dir=final_root,
        output_md=final_root / "comparison_report.md",
    )


def _episode_key_from_trace(root: Path, trace: Path) -> str:
    parts = trace.relative_to(root).parts
    seed = trace.name.removeprefix("episode_").removesuffix("_trace.jsonl")
    return "_".join((parts[1], parts[2], parts[3], parts[4], seed))


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
    environment_factory: EvaluationEnvironmentFactory,
    policy_factory: EvaluationPolicyFactory,
    frame_provider: RgbFrameProvider,
    selection_scores: tuple[CheckpointScore, ...] | None,
    checkpoint_reader: CheckpointReader = validate_ppo_checkpoint_archive,
    event_reader: TrainingEventReader = extract_training_metrics_from_event_sources,
) -> Path:
    """Build, verify, and atomically publish a complete Phase 6 artifact bundle."""

    final_destination = validate_absent_destination(
        destination,
        source_roots=tuple(
            Path(binding.training_run_dir) for binding in evaluation_config.ppo_run_bindings
        ),
    )
    source_workspace: EvaluationWorkspace | None = None
    final_workspace: EvaluationWorkspace | None = None
    try:
        configs, candidates = _validated_inputs(
            evaluation_config=evaluation_config,
            plan_path=Path(plan_path),
            run_plan=run_plan,
            method_configs=method_configs,
            method_profiles=method_profiles,
            authenticated_checkpoints=authenticated_checkpoints,
            checkpoint_reader=checkpoint_reader,
            selection_scores=selection_scores,
            smoke=smoke,
        )
        source_workspace = EvaluationWorkspace.stage(final_destination)
        source_root = source_workspace.path
        _write_source_provenance(
            source_root,
            evaluation_config=evaluation_config,
            plan_path=Path(plan_path),
            method_configs=method_configs,
            method_profiles=method_profiles,
            cli_overlays=cli_overlays,
        )
        if selection_scores is None:
            write_unselected_smoke_checkpoint_artifacts(source_root, tuple(candidates.values()))
        else:
            write_selection_artifacts(source_root, selection_scores)
        event_sources = _copy_tensorboard_sources(source_root, evaluation_config)
        points = event_reader(event_sources, smoke=smoke)
        _validate_training_points(points, evaluation_config)
        write_training_metrics_csv(
            source_root / "metrics" / "train_metrics.csv", points, smoke=smoke
        )
        metric_records = _write_online_outputs(
            source_root,
            evaluation_config=evaluation_config,
            run_plan=run_plan,
            configs=configs,
            candidates=candidates,
            environment_factory=environment_factory,
            policy_factory=policy_factory,
            frame_provider=frame_provider,
        )
        validate_matched_episodes(metric_records)
        comparison_rows = build_comparison_rows(metric_records)
        write_eval_metrics_csv(source_root / "metrics" / "eval_metrics.csv", metric_records)
        write_comparison_csv(source_root / "metrics" / "comparison.csv", comparison_rows)
        source_workspace.write_manifest()
        _verify_bundle(source_root)

        final_workspace = EvaluationWorkspace.stage(final_destination)
        _copy_verified_source(source_root, final_workspace.path)
        _render_final_outputs(source_root, final_workspace.path, evaluation_config)
        final_workspace.write_manifest()
        _verify_bundle(final_workspace.path)
        _cleanup_workspace(source_workspace)
        source_workspace = None
        published = final_workspace.publish()
        return published
    except BaseException as error:
        for label, workspace in (
            ("final", final_workspace),
            ("source", source_workspace),
        ):
            try:
                _cleanup_workspace(workspace)
            except BaseException as cleanup_error:
                error.add_note(f"{label} private workspace cleanup also failed: {cleanup_error}")
        raise


__all__ = [
    "EvaluationEnvironmentFactory",
    "EvaluationPolicyFactory",
    "RgbFrameProvider",
    "TrainingEventReader",
    "run_evaluation_bundle",
]
