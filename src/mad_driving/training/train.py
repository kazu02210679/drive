"""PPO training lifecycle, vector environments, and artifacts."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from numbers import Integral
from pathlib import Path
from time import monotonic as _monotonic
from typing import Any, Protocol, TypeVar

import gymnasium as gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mad_driving.config.models import AppConfig
from mad_driving.envs.multi_agent_speed_env import MultiAgentSpeedEnv
from mad_driving.scenarios import EnvironmentRole
from mad_driving.training.callbacks import (
    CurriculumCheckpointCallback,
    CurriculumEvalCallback,
    RewardComponentsCallback,
)
from mad_driving.training.curriculum import (
    CURRICULUM_STATE_FILENAME,
    CurriculumController,
    CurriculumState,
    write_checkpoint_curriculum_state,
    write_curriculum_state,
)
from mad_driving.training.episode_seeds import (
    EpisodeSeedArtifactDescriptor,
    EpisodeSeedRecordingWrapper,
    summarize_episode_seed_artifacts,
)
from mad_driving.training.metadata import (
    ResumeMetadata,
    RunMetadata,
    checkpoint_curriculum_artifact_inventory,
    curriculum_state_artifact,
    resolve_resume_source,
    validate_resume_contract,
    write_run_metadata,
)
from mad_driving.training.ownership import RunDirectoryOwnership


class EnvironmentFactory(Protocol):
    def __call__(
        self,
        config: AppConfig,
        *,
        role: EnvironmentRole,
        worker_index: int,
    ) -> gym.Env[Any, Any]: ...


class _VectorEnvCleanupError(RuntimeError):
    """Raised when failed subprocess workers cannot be proven stopped."""


class _CheckpointPromotionRollbackError(RuntimeError):
    """Raised when canonical checkpoints could not be fully restored."""


FactoryResult = TypeVar("FactoryResult")
VecEnvFactory = Callable[[list[Callable[[], gym.Env[Any, Any]]]], Any]
GenericFactory = Callable[..., FactoryResult]
_STAGING_PREFIX = ".training-"
_VECTOR_CLOSE_AUDIT_MARKER = "_mad_driving_close_audit_complete"
_GRACEFUL_PROCESS_JOIN_TIMEOUT_SECONDS = 5.0
_GRACEFUL_JOIN_PHASE_SECONDS = 3.0
_TERMINATE_JOIN_PHASE_SECONDS = 1.0
_KILL_JOIN_PHASE_SECONDS = 1.0


@dataclass(frozen=True)
class TrainingResult:
    """Immutable paths and timestep count produced by one training run."""

    run_dir: Path
    final_checkpoint: Path
    best_checkpoint: Path
    timesteps: int


def require_empty_run_directory(path: Path) -> None:
    """Reject destinations that could contain user-owned artifacts."""

    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Run directory is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory is non-empty: {path}")


def _default_env_factory(
    config: AppConfig,
    *,
    role: EnvironmentRole,
    worker_index: int,
) -> gym.Env[Any, Any]:
    return MultiAgentSpeedEnv(config, role=role, worker_index=worker_index)


@dataclass
class _OwnedEnvironments:
    config: AppConfig
    factory: EnvironmentFactory
    workspace: Path
    created: list[gym.Env[Any, Any]]

    @classmethod
    def create(
        cls,
        config: AppConfig,
        factory: EnvironmentFactory,
        workspace: Path,
    ) -> _OwnedEnvironments:
        return cls(config=config, factory=factory, workspace=workspace, created=[])

    def new(
        self,
        config: AppConfig,
        *,
        role: EnvironmentRole,
        worker_index: int,
    ) -> gym.Env[Any, Any]:
        environment = self.factory(
            config,
            role=role,
            worker_index=worker_index,
        )
        try:
            wrapped = EpisodeSeedRecordingWrapper(
                environment,
                workspace=self.workspace,
                role=role,
                worker_index=worker_index,
            )
        except BaseException as construction_error:
            try:
                environment.close()
            except Exception as cleanup_error:
                construction_error.add_note(f"Cleanup also failed: {cleanup_error}")
            raise
        self.created.append(wrapped)
        return wrapped

    def thunks(
        self,
        role: EnvironmentRole,
        count: int,
    ) -> list[Callable[[], gym.Env[Any, Any]]]:
        return [
            partial(
                self.new,
                self.config.model_copy(deep=True),
                role=role,
                worker_index=worker_index,
            )
            for worker_index in range(count)
        ]

    def transfer(self) -> None:
        """Transfer underlying environment ownership to a constructed vector env."""

        self.created.clear()

    def close(self) -> None:
        environments = self.created
        self.created = []
        errors: list[Exception] = []
        for environment in environments:
            try:
                environment.close()
            except Exception as error:
                errors.append(error)
        if errors:
            details = "; ".join(str(error) for error in errors)
            combined = RuntimeError(f"Owned environment cleanup failed: {details}")
            for additional_error in errors[1:]:
                combined.add_note(f"Additional owner cleanup failure: {additional_error}")
            raise combined from errors[0]


def _close_model_logger(model: object | None) -> None:
    logger = getattr(model, "logger", None)
    for output_format in tuple(getattr(logger, "output_formats", ())):
        close = getattr(output_format, "close", None)
        if callable(close):
            close()


def _collect_episode_seed_artifact_descriptors(
    vector_env: object,
    *,
    expected_environments: Sequence[tuple[EnvironmentRole, int]],
) -> tuple[EpisodeSeedArtifactDescriptor, ...]:
    """Fetch live writer identities over the VecEnv control channel."""

    get_attr = getattr(vector_env, "get_attr", None)
    if not callable(get_attr):
        raise RuntimeError("Vector environment cannot report episode seed artifact identities")
    try:
        raw_descriptors = get_attr("episode_seed_artifact_descriptor")
    except Exception as exc:
        raise RuntimeError(
            "Vector environment failed to report episode seed artifact identities"
        ) from exc
    if not isinstance(raw_descriptors, list | tuple):
        raise RuntimeError("Vector environment returned malformed seed artifact identities")
    raw_descriptor_values = tuple(raw_descriptors)
    if len(raw_descriptor_values) != len(expected_environments):
        raise RuntimeError("Vector environment returned an incomplete seed artifact inventory")
    descriptors: list[EpisodeSeedArtifactDescriptor] = []
    for descriptor in raw_descriptor_values:
        if not isinstance(descriptor, EpisodeSeedArtifactDescriptor):
            raise RuntimeError("Vector environment returned a malformed seed artifact descriptor")
        descriptors.append(descriptor)
    if len(set(descriptors)) != len(descriptors):
        raise RuntimeError("Vector environment returned duplicate seed artifact descriptors")
    for descriptor, expected_environment in zip(
        descriptors,
        expected_environments,
        strict=True,
    ):
        if (descriptor.role, descriptor.worker_index) != expected_environment:
            raise RuntimeError("Vector environment returned a mismatched seed artifact descriptor")
    return tuple(descriptors)


def _process_is_alive(process: object) -> bool:
    try:
        return bool(process.is_alive())  # type: ignore[attr-defined]
    except Exception:
        return True


def _attempt_process_operation(
    operation: Callable[..., object],
    *args: object,
) -> Exception | None:
    try:
        operation(*args)
    except Exception as exc:
        return exc
    return None


def _stop_processes(
    processes: list[object],
    *,
    graceful_join: bool,
) -> tuple[bool, bool, tuple[Exception, ...]]:
    """Escalate through join, terminate, and kill despite intermediate failures."""

    escalated = False
    operation_errors: list[Exception] = []
    started_at = _monotonic()
    deadline = started_at + _GRACEFUL_PROCESS_JOIN_TIMEOUT_SECONDS

    def join_for(candidates: list[object], phase_seconds: float) -> None:
        joinable = [
            (process, join)
            for process in candidates
            if callable(join := getattr(process, "join", None))
        ]
        if not joinable:
            return
        phase_deadline = min(deadline, _monotonic() + phase_seconds)
        for index, (_process, join) in enumerate(joinable):
            phase_remaining = max(0.0, phase_deadline - _monotonic())
            remaining_candidates = len(joinable) - index
            timeout = phase_remaining / remaining_candidates
            if error := _attempt_process_operation(join, timeout):
                operation_errors.append(error)

    if graceful_join:
        join_for(processes, _GRACEFUL_JOIN_PHASE_SECONDS)

    terminated_processes: list[object] = []
    for process in processes:
        if _process_is_alive(process):
            escalated = True
            terminated_processes.append(process)
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                if error := _attempt_process_operation(terminate):
                    operation_errors.append(error)

    join_for(terminated_processes, _TERMINATE_JOIN_PHASE_SECONDS)

    killed_processes: list[object] = []
    for process in processes:
        if _process_is_alive(process):
            escalated = True
            killed_processes.append(process)
            kill = getattr(process, "kill", None)
            if callable(kill):
                if error := _attempt_process_operation(kill):
                    operation_errors.append(error)

    join_for(killed_processes, _KILL_JOIN_PHASE_SECONDS)
    return (
        any(_process_is_alive(process) for process in processes),
        escalated,
        tuple(operation_errors),
    )


def _close_vector_env(resource: object | None) -> None:
    """Close a vector env and prove subprocess workers have stopped."""

    if resource is None:
        return
    if getattr(resource, _VECTOR_CLOSE_AUDIT_MARKER, False):
        return
    processes = getattr(resource, "processes", None)
    if not isinstance(processes, list):
        close = getattr(resource, "close", None)
        if callable(close):
            close()
        try:
            setattr(resource, _VECTOR_CLOSE_AUDIT_MARKER, True)
        except Exception as exc:
            raise _VectorEnvCleanupError(
                "Vector environment close idempotence could not be established"
            ) from exc
        return

    remotes = tuple(getattr(resource, "remotes", ()))
    cleanup_errors: list[Exception] = []
    if not getattr(resource, "closed", False) and not getattr(resource, "waiting", False):
        for remote in remotes:
            send = getattr(remote, "send", None)
            if callable(send):
                try:
                    send(("close", None))
                except Exception as error:
                    cleanup_errors.append(
                        RuntimeError(f"Subprocess close request could not be confirmed: {error}")
                    )

    workers_alive, escalated, process_operation_errors = _stop_processes(
        processes,
        graceful_join=True,
    )
    for operation_error in process_operation_errors:
        cleanup_errors.append(
            RuntimeError(f"Subprocess shutdown operation failed: {operation_error}")
        )
    if escalated:
        cleanup_errors.append(
            _VectorEnvCleanupError("Subprocess worker required terminate/kill escalation")
        )
    exit_codes_confirmed = True
    for worker_index, process in enumerate(processes):
        exitcode = getattr(process, "exitcode", None)
        if exitcode is None:
            exit_codes_confirmed = False
            cleanup_errors.append(
                _VectorEnvCleanupError(
                    f"Subprocess worker {worker_index} shutdown exit code is unconfirmed"
                )
            )
        elif isinstance(exitcode, bool) or not isinstance(exitcode, Integral):
            exit_codes_confirmed = False
            cleanup_errors.append(
                _VectorEnvCleanupError(
                    f"Subprocess worker {worker_index} returned malformed exit code"
                )
            )
        elif int(exitcode) != 0:
            cleanup_errors.append(
                _VectorEnvCleanupError(
                    f"Subprocess worker {worker_index} exited with exit code {int(exitcode)}"
                )
            )

    for remote in remotes:
        try:
            remote.close()
        except Exception as error:
            cleanup_errors.append(
                RuntimeError(f"Subprocess remote cleanup could not be confirmed: {error}")
            )
    try:
        resource.closed = True  # type: ignore[attr-defined]
    except Exception as error:
        cleanup_errors.append(
            RuntimeError(f"Subprocess close state could not be confirmed: {error}")
        )

    if workers_alive:
        cleanup_errors.append(
            _VectorEnvCleanupError("Subprocess worker cleanup could not be confirmed")
        )
    elif exit_codes_confirmed:
        try:
            setattr(resource, _VECTOR_CLOSE_AUDIT_MARKER, True)
        except Exception as error:
            cleanup_errors.append(
                _VectorEnvCleanupError(
                    f"Vector environment close audit could not be recorded: {error}"
                )
            )
    if cleanup_errors:
        details = "; ".join(str(error) for error in cleanup_errors)
        raise _VectorEnvCleanupError(details) from cleanup_errors[0]


def _cleanup_partial_vector_env(
    vector_env: object,
    owner: _OwnedEnvironments,
) -> None:
    """Release a vector env retained from a failed class constructor."""

    processes = getattr(vector_env, "processes", None)
    if isinstance(processes, list):
        remote_close_failed = False
        for remote_group_name in ("remotes", "work_remotes"):
            for remote in getattr(vector_env, remote_group_name, ()):
                try:
                    remote.close()
                except Exception:
                    remote_close_failed = True
        workers_alive, _, process_operation_errors = _stop_processes(
            processes,
            graceful_join=False,
        )
        owner.close()
        if workers_alive:
            raise _VectorEnvCleanupError("Subprocess worker cleanup could not be confirmed")
        if process_operation_errors:
            details = "; ".join(str(error) for error in process_operation_errors)
            raise _VectorEnvCleanupError(
                f"Subprocess shutdown operation failed: {details}"
            ) from process_operation_errors[0]
        if remote_close_failed:
            raise _VectorEnvCleanupError("Subprocess remote cleanup could not be confirmed")
        return

    close = getattr(vector_env, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            owner.close()
            raise _VectorEnvCleanupError(
                "Partial vector environment cleanup could not be confirmed"
            ) from exc
        owner.transfer()
        return
    owner.close()


def _construct_vector_env(
    factory: VecEnvFactory,
    env_fns: list[Callable[[], gym.Env[Any, Any]]],
    owner: _OwnedEnvironments,
) -> Any:
    """Construct while retaining class instances long enough to clean failed initialization."""

    if isinstance(factory, type):
        factory_class: Any = factory
        instance = factory_class.__new__(factory_class)
        try:
            factory_class.__init__(instance, env_fns)
        except BaseException as construction_error:
            try:
                _cleanup_partial_vector_env(instance, owner)
            except Exception as cleanup_error:
                construction_error.add_note(f"Cleanup also failed: {cleanup_error}")
            raise
        owner.transfer()
        return instance
    vector_env = factory(env_fns)
    owner.transfer()
    return vector_env


def _build_train_env(
    config: AppConfig,
    owner: _OwnedEnvironments,
    *,
    dummy_vec_env_factory: VecEnvFactory,
    subproc_vec_env_factory: VecEnvFactory,
) -> Any:
    count = config.training.num_envs
    if count == 1:
        return _construct_vector_env(
            dummy_vec_env_factory,
            owner.thunks("train", 1),
            owner,
        )
    return _construct_vector_env(
        subproc_vec_env_factory,
        owner.thunks("train", count),
        owner,
    )


def _build_eval_env(
    config: AppConfig,
    owner: _OwnedEnvironments,
    *,
    dummy_vec_env_factory: VecEnvFactory,
    subproc_vec_env_factory: VecEnvFactory,
) -> Any:
    factory = subproc_vec_env_factory if config.training.num_envs == 1 else dummy_vec_env_factory
    return _construct_vector_env(
        factory,
        owner.thunks("validation", 1),
        owner,
    )


def _scaled_frequency(interval_steps: int, num_envs: int) -> int:
    return max(interval_steps // num_envs, 1)


def _initial_curriculum_state(config: AppConfig) -> CurriculumState:
    curriculum = config.scenarios.curriculum
    level = curriculum.fixed_level if curriculum.mode == "fixed" else curriculum.initial_level
    return CurriculumState(level=level, consecutive_passes=0, evaluations=0)


def _broadcast_difficulty_level(vector_env: object, level: int) -> None:
    env_method = getattr(vector_env, "env_method", None)
    if not callable(env_method):
        raise RuntimeError("Vector environment cannot restore the curriculum difficulty level")
    env_method("set_difficulty_level", level)


def _write_resolved_config(config: AppConfig, destination: Path) -> None:
    serialized = yaml.safe_dump(
        config.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
    )
    destination.write_text(serialized, encoding="utf-8")


def _validated_staging_path(staging_dir: Path, checkpoints_dir: Path) -> Path:
    checkpoints_root = checkpoints_dir.resolve(strict=True)
    staging_root = staging_dir.resolve(strict=True)
    if staging_root.parent != checkpoints_root or not staging_root.name.startswith(_STAGING_PREFIX):
        raise RuntimeError(f"Refusing checkpoint staging operation outside: {checkpoints_root}")
    return staging_root


def _create_checkpoint_staging(checkpoints_dir: Path) -> Path:
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=_STAGING_PREFIX,
            dir=checkpoints_dir,
        )
    )
    _validated_staging_path(staging_dir, checkpoints_dir)
    return staging_dir


def _promote_checkpoint_artifacts(staging_dir: Path, checkpoints_dir: Path) -> None:
    staging_root = _validated_staging_path(staging_dir, checkpoints_dir)
    artifacts = list(staging_root.iterdir())
    if any(not artifact.is_file() for artifact in artifacts):
        raise RuntimeError(f"Checkpoint staging contains a non-file artifact: {staging_root}")
    artifacts.sort(
        key=lambda artifact: (
            artifact.name in {"best_model.zip", "final_model.zip"},
            artifact.name,
        )
    )
    backup_dir = Path(tempfile.mkdtemp(prefix=".promotion-backups-", dir=staging_root))
    backups: dict[Path, Path] = {}
    new_destinations: list[Path] = []
    for artifact in artifacts:
        destination = checkpoints_dir / artifact.name
        if destination.exists():
            if not destination.is_file():
                raise RuntimeError(f"Checkpoint destination is not a file: {destination}")
            backup = backup_dir / artifact.name
            shutil.copy2(destination, backup)
            backups[destination] = backup
        else:
            new_destinations.append(destination)

    try:
        for artifact in artifacts:
            os.replace(artifact, checkpoints_dir / artifact.name)
    except Exception as promotion_error:
        rollback_errors: list[Exception] = []
        for destination, backup in backups.items():
            try:
                os.replace(backup, destination)
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
        for destination in new_destinations:
            try:
                destination.unlink(missing_ok=True)
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise _CheckpointPromotionRollbackError(
                "Checkpoint promotion failed and rollback could not be completed; "
                f"recoverable backups remain in {backup_dir}"
            ) from promotion_error
        raise


def _cleanup_checkpoint_staging(staging_dir: Path, checkpoints_dir: Path) -> None:
    if not staging_dir.exists():
        return
    staging_root = _validated_staging_path(staging_dir, checkpoints_dir)
    shutil.rmtree(staging_root)


def run_training(
    config: AppConfig,
    *,
    smoke: bool,
    run_dir: str | Path,
    resume_from: str | Path | None = None,
    require_absent_run_dir: bool = False,
    env_factory: EnvironmentFactory = _default_env_factory,
    ppo_factory: GenericFactory[Any] = PPO,
    dummy_vec_env_factory: VecEnvFactory = DummyVecEnv,
    subproc_vec_env_factory: VecEnvFactory = SubprocVecEnv,
    checkpoint_callback_factory: GenericFactory[Any] = CurriculumCheckpointCallback,
    eval_callback_factory: GenericFactory[Any] = CurriculumEvalCallback,
    reward_callback_factory: GenericFactory[Any] = RewardComponentsCallback,
) -> TrainingResult:
    """Train or resume one PPO policy and close every environment on exit."""

    destination = Path(run_dir)
    require_empty_run_directory(destination)
    resume_path = None if resume_from is None else Path(resume_from)
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    resume_source = None if resume_path is None else resolve_resume_source(resume_path, config)
    curriculum_state = (
        _initial_curriculum_state(config)
        if resume_source is None
        else resume_source.curriculum_state
    )
    curriculum_controller = CurriculumController(
        config.scenarios.curriculum,
        curriculum_state,
    )
    if resume_source is not None:
        canonical_destination = destination.resolve()
        if (
            canonical_destination == resume_source.run_dir
            or resume_source.run_dir in canonical_destination.parents
        ):
            raise ValueError(
                f"Resume destination must be separate from the source run: {destination}"
            )

    ownership = RunDirectoryOwnership.acquire(
        destination,
        require_absent=require_absent_run_dir,
    )
    workspace = ownership.workspace
    curriculum_state_path = workspace / CURRICULUM_STATE_FILENAME
    checkpoints_dir = workspace / "checkpoints"
    tensorboard_dir = workspace / "tensorboard"
    requested_timesteps = (
        config.training.smoke_timesteps if smoke else config.training.total_timesteps
    )
    final_checkpoint = checkpoints_dir / "final_model.zip"
    best_checkpoint = checkpoints_dir / "best_model.zip"
    published_final_checkpoint = destination / "checkpoints" / final_checkpoint.name
    published_best_checkpoint = destination / "checkpoints" / best_checkpoint.name
    staging_dir: Path | None = None
    cleanup_staging = True

    train_owner = _OwnedEnvironments.create(config, env_factory, workspace)
    eval_owner = _OwnedEnvironments.create(config, env_factory, workspace)
    train_env: Any = None
    eval_env: Any = None
    model: Any = None
    model_logger_closed = False
    primary_error: BaseException | None = None
    try:
        write_curriculum_state(curriculum_state, curriculum_state_path)
        train_env = _build_train_env(
            config,
            train_owner,
            dummy_vec_env_factory=dummy_vec_env_factory,
            subproc_vec_env_factory=subproc_vec_env_factory,
        )
        eval_env = _build_eval_env(
            config,
            eval_owner,
            dummy_vec_env_factory=dummy_vec_env_factory,
            subproc_vec_env_factory=subproc_vec_env_factory,
        )
        if resume_source is not None:
            _broadcast_difficulty_level(train_env, curriculum_state.level)
            _broadcast_difficulty_level(eval_env, curriculum_state.level)

        if resume_source is None:
            model = ppo_factory(
                config.training.policy,
                train_env,
                learning_rate=config.training.learning_rate,
                n_steps=config.training.n_steps,
                batch_size=config.training.batch_size,
                n_epochs=config.training.n_epochs,
                gamma=config.training.gamma,
                gae_lambda=config.training.gae_lambda,
                clip_range=config.training.clip_range,
                ent_coef=config.training.ent_coef,
                vf_coef=config.training.vf_coef,
                max_grad_norm=config.training.max_grad_norm,
                seed=config.training.seed,
                tensorboard_log=str(tensorboard_dir),
                device="cpu",
            )
        else:
            model = ppo_factory.load(  # type: ignore[attr-defined]
                io.BytesIO(resume_source.checkpoint_bytes),
                env=train_env,
                device="cpu",
                tensorboard_log=str(tensorboard_dir),
            )
            validate_resume_contract(model, config, resume_source.metadata)

        raw_start_timesteps = model.num_timesteps
        if (
            isinstance(raw_start_timesteps, bool)
            or not isinstance(raw_start_timesteps, Integral)
            or raw_start_timesteps < 0
        ):
            raise ValueError("Resume num_timesteps must be a non-negative non-bool integer")
        start_timesteps = int(raw_start_timesteps)
        resume_metadata = None
        if resume_source is not None:
            resume_metadata = ResumeMetadata(
                parent_checkpoint_path=str(resume_source.checkpoint),
                parent_checkpoint_sha256=resume_source.checkpoint_sha256,
                parent_run_dir=str(resume_source.run_dir),
                parent_config=resume_source.resolved_config,
                config_diff=resume_source.config_diff,
                start_num_timesteps=start_timesteps,
            )

        checkpoints_dir.mkdir(parents=True)
        tensorboard_dir.mkdir()
        _write_resolved_config(config, workspace / "config_resolved.yaml")
        write_run_metadata(
            RunMetadata(
                resolved_config=config.model_dump(mode="json"),
                resume=resume_metadata,
                curriculum_state=curriculum_state_artifact(
                    curriculum_state_path,
                    curriculum_controller.state,
                ),
            ),
            workspace / "run_metadata.json",
        )
        staging_dir = _create_checkpoint_staging(checkpoints_dir)
        staged_final_checkpoint = staging_dir / final_checkpoint.name
        staged_best_checkpoint = staging_dir / best_checkpoint.name
        checkpoint_callback = checkpoint_callback_factory(
            controller=curriculum_controller,
            save_freq=_scaled_frequency(
                config.training.checkpoint_interval_steps,
                config.training.num_envs,
            ),
            save_path=str(staging_dir),
            name_prefix="ppo_checkpoint",
        )
        eval_callback = eval_callback_factory(
            eval_env=eval_env,
            validation_episode_seed=config.seed,
            controller=curriculum_controller,
            curriculum_state_path=curriculum_state_path,
            best_model_save_path=str(staging_dir),
            eval_freq=_scaled_frequency(
                min(config.training.eval_interval_steps, requested_timesteps),
                config.training.num_envs,
            ),
            n_eval_episodes=config.training.eval_episodes,
            deterministic=True,
            render=False,
        )
        reward_callback = reward_callback_factory()
        callbacks = [reward_callback, checkpoint_callback, eval_callback]
        model.learn(
            total_timesteps=requested_timesteps,
            callback=callbacks,
            reset_num_timesteps=resume_source is None,
        )
        model.save(staged_final_checkpoint)
        if not staged_final_checkpoint.is_file():
            raise FileNotFoundError(
                f"Final checkpoint was not produced: {staged_final_checkpoint}"
            )
        write_checkpoint_curriculum_state(
            curriculum_controller.state,
            staged_final_checkpoint,
        )
        trusted_seed_descriptors = (
            *_collect_episode_seed_artifact_descriptors(
                train_env,
                expected_environments=tuple(
                    ("train", worker_index) for worker_index in range(config.training.num_envs)
                ),
            ),
            *_collect_episode_seed_artifact_descriptors(
                eval_env,
                expected_environments=(("validation", 0),),
            ),
        )
        if not staged_best_checkpoint.is_file():
            raise FileNotFoundError(f"Best checkpoint was not produced: {staged_best_checkpoint}")
        model_logger_closed = True
        _close_model_logger(model)
        _promote_checkpoint_artifacts(staging_dir, checkpoints_dir)
        checkpoint_curriculum_artifacts = checkpoint_curriculum_artifact_inventory(
            checkpoints_dir
        )
        try:
            _cleanup_checkpoint_staging(staging_dir, checkpoints_dir)
        except Exception as exc:
            cleanup_staging = False
            raise RuntimeError(f"Training resource cleanup failed: {exc}") from exc
        staging_dir = None
        timesteps = int(model.num_timesteps) - start_timesteps
        closing_eval_env = eval_env
        eval_env = None
        _close_vector_env(closing_eval_env)
        closing_train_env = train_env
        train_env = None
        _close_vector_env(closing_train_env)
        for owner in (eval_owner, train_owner):
            owner.close()
        episode_seed_artifacts = summarize_episode_seed_artifacts(
            workspace,
            expected_descriptors=trusted_seed_descriptors,
        )
        write_curriculum_state(curriculum_controller.state, curriculum_state_path)
        write_run_metadata(
            RunMetadata(
                resolved_config=config.model_dump(mode="json"),
                resume=resume_metadata,
                curriculum_state=curriculum_state_artifact(
                    curriculum_state_path,
                    curriculum_controller.state,
                ),
                checkpoint_curriculum_artifacts=checkpoint_curriculum_artifacts,
                episode_seed_artifacts=episode_seed_artifacts,
            ),
            workspace / "run_metadata.json",
        )
        ownership.publish()
        return TrainingResult(
            run_dir=destination,
            final_checkpoint=published_final_checkpoint,
            best_checkpoint=published_best_checkpoint,
            timesteps=timesteps,
        )
    except BaseException as exc:
        primary_error = exc
        if isinstance(exc, _CheckpointPromotionRollbackError):
            cleanup_staging = False
        raise
    finally:
        cleanup_errors: list[Exception] = []
        if not model_logger_closed:
            model_logger_closed = True
            try:
                _close_model_logger(model)
            except Exception as exc:
                cleanup_errors.append(exc)
        for vector_env in (eval_env, train_env):
            try:
                _close_vector_env(vector_env)
            except Exception as exc:
                cleanup_errors.append(exc)
        for owner in (eval_owner, train_owner):
            try:
                owner.close()
            except Exception as exc:
                cleanup_errors.append(exc)
        if cleanup_staging and staging_dir is not None:
            try:
                _cleanup_checkpoint_staging(staging_dir, checkpoints_dir)
            except Exception as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(str(error) for error in cleanup_errors)
            cleanup_error = RuntimeError(f"Training resource cleanup failed: {details}")
            if primary_error is not None:
                primary_error.add_note(f"Cleanup also failed: {details}")
        else:
            cleanup_error = None
        try:
            ownership.release()
        except Exception as exc:
            if primary_error is not None:
                primary_error.add_note(f"Ownership cleanup also failed: {exc}")
            elif cleanup_error is None:
                raise
            else:
                cleanup_error.add_note(f"Ownership cleanup also failed: {exc}")
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error from cleanup_errors[0]
