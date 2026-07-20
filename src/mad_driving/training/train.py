"""PPO training lifecycle, vector environments, and artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeVar

import gymnasium as gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mad_driving.config.models import AppConfig
from mad_driving.envs.multi_agent_speed_env import MultiAgentSpeedEnv
from mad_driving.training.callbacks import RewardComponentsCallback


class EnvironmentFactory(Protocol):
    def __call__(self, config: AppConfig) -> gym.Env[Any, Any]: ...


FactoryResult = TypeVar("FactoryResult")
VecEnvFactory = Callable[[list[Callable[[], gym.Env[Any, Any]]]], Any]
GenericFactory = Callable[..., FactoryResult]


@dataclass(frozen=True)
class TrainingResult:
    """Immutable paths and timestep count produced by one training run."""

    run_dir: Path
    final_checkpoint: Path
    best_checkpoint: Path
    timesteps: int


def _default_env_factory(config: AppConfig) -> gym.Env[Any, Any]:
    return MultiAgentSpeedEnv(config)


@dataclass
class _OwnedEnvironments:
    config: AppConfig
    factory: EnvironmentFactory
    created: list[gym.Env[Any, Any]]

    @classmethod
    def create(cls, config: AppConfig, factory: EnvironmentFactory) -> _OwnedEnvironments:
        return cls(config=config, factory=factory, created=[])

    def new(self) -> gym.Env[Any, Any]:
        environment = self.factory(self.config)
        self.created.append(environment)
        return environment

    def thunks(self, count: int) -> list[Callable[[], gym.Env[Any, Any]]]:
        return [partial(self.new) for _ in range(count)]

    def close(self) -> None:
        for environment in self.created:
            _safe_close(environment)


def _safe_close(resource: object | None) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _cleanup_partial_vector_env(vector_env: object) -> None:
    """Release a vector env retained from a failed class constructor."""

    processes = getattr(vector_env, "processes", None)
    if isinstance(processes, list):
        for remote_group_name in ("remotes", "work_remotes"):
            for remote in getattr(vector_env, remote_group_name, ()):
                _safe_close(remote)
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=1.0)
            except Exception:
                pass
        return
    _safe_close(vector_env)


def _construct_vector_env(
    factory: VecEnvFactory,
    env_fns: list[Callable[[], gym.Env[Any, Any]]],
) -> Any:
    """Construct while retaining class instances long enough to clean failed initialization."""

    if isinstance(factory, type):
        factory_class: Any = factory
        instance = factory_class.__new__(factory_class)
        try:
            factory_class.__init__(instance, env_fns)
        except Exception:
            _cleanup_partial_vector_env(instance)
            raise
        return instance
    return factory(env_fns)


def _build_train_env(
    config: AppConfig,
    owner: _OwnedEnvironments,
    *,
    dummy_vec_env_factory: VecEnvFactory,
    subproc_vec_env_factory: VecEnvFactory,
) -> Any:
    count = config.training.num_envs
    if count == 1:
        return _construct_vector_env(dummy_vec_env_factory, owner.thunks(1))
    try:
        return _construct_vector_env(subproc_vec_env_factory, owner.thunks(count))
    except Exception:
        owner.close()
        owner.created.clear()
        return _construct_vector_env(dummy_vec_env_factory, owner.thunks(count))


def _scaled_frequency(interval_steps: int, num_envs: int) -> int:
    return max(interval_steps // num_envs, 1)


def _write_resolved_config(config: AppConfig, destination: Path) -> None:
    serialized = yaml.safe_dump(
        config.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
    )
    destination.write_text(serialized, encoding="utf-8")


def run_training(
    config: AppConfig,
    *,
    smoke: bool,
    run_dir: str | Path,
    resume_from: str | Path | None = None,
    env_factory: EnvironmentFactory = _default_env_factory,
    ppo_factory: GenericFactory[Any] = PPO,
    dummy_vec_env_factory: VecEnvFactory = DummyVecEnv,
    subproc_vec_env_factory: VecEnvFactory = SubprocVecEnv,
    checkpoint_callback_factory: GenericFactory[Any] = CheckpointCallback,
    eval_callback_factory: GenericFactory[Any] = EvalCallback,
    reward_callback_factory: GenericFactory[Any] = RewardComponentsCallback,
) -> TrainingResult:
    """Train or resume one PPO policy and close every environment on exit."""

    destination = Path(run_dir)
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Run directory is not a directory: {destination}")
    resume_path = None if resume_from is None else Path(resume_from)
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoints_dir = destination / "checkpoints"
    tensorboard_dir = destination / "tensorboard"
    evaluation_dir = destination / "evaluation"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(config, destination / "config_resolved.yaml")

    train_owner = _OwnedEnvironments.create(config, env_factory)
    eval_owner = _OwnedEnvironments.create(config, env_factory)
    train_env: Any = None
    eval_env: Any = None
    try:
        train_env = _build_train_env(
            config,
            train_owner,
            dummy_vec_env_factory=dummy_vec_env_factory,
            subproc_vec_env_factory=subproc_vec_env_factory,
        )
        eval_env = _construct_vector_env(dummy_vec_env_factory, eval_owner.thunks(1))

        checkpoint_callback = checkpoint_callback_factory(
            save_freq=_scaled_frequency(
                config.training.checkpoint_interval_steps,
                config.training.num_envs,
            ),
            save_path=str(checkpoints_dir),
            name_prefix="ppo_checkpoint",
        )
        eval_callback = eval_callback_factory(
            eval_env=eval_env,
            best_model_save_path=str(checkpoints_dir),
            log_path=str(evaluation_dir),
            eval_freq=_scaled_frequency(
                config.training.eval_interval_steps,
                config.training.num_envs,
            ),
            n_eval_episodes=config.training.eval_episodes,
            deterministic=True,
            render=False,
        )
        reward_callback = reward_callback_factory()

        if resume_path is None:
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
            model = ppo_factory.load(resume_path, env=train_env, device="cpu")  # type: ignore[attr-defined]

        timesteps = config.training.smoke_timesteps if smoke else config.training.total_timesteps
        model.learn(
            total_timesteps=timesteps,
            callback=[reward_callback, checkpoint_callback, eval_callback],
            reset_num_timesteps=resume_path is None,
        )
        final_checkpoint = checkpoints_dir / "final_model.zip"
        model.save(final_checkpoint)
        return TrainingResult(
            run_dir=destination,
            final_checkpoint=final_checkpoint,
            best_checkpoint=checkpoints_dir / "best_model.zip",
            timesteps=timesteps,
        )
    finally:
        _safe_close(eval_env)
        _safe_close(train_env)
        eval_owner.close()
        train_owner.close()
