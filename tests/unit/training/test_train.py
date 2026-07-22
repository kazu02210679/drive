import hashlib
import io
import json
import math
import os
import threading
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cloudpickle
import gymnasium as gym
import numpy as np
import pytest
import yaml
from gymnasium import spaces
from stable_baselines3.common.utils import ConstantSchedule, FloatSchedule

from mad_driving.config.models import AppConfig
from mad_driving.methods import get_method_profile
from mad_driving.scenarios import EnvironmentRole
from mad_driving.training import RunMetadata, sha256_file
from mad_driving.training import metadata as metadata_module
from mad_driving.training import ownership as ownership_module
from mad_driving.training import train as train_module
from mad_driving.training.curriculum import (
    CurriculumState,
    checkpoint_curriculum_sidecar_path,
    read_curriculum_state,
    write_checkpoint_curriculum_state,
    write_curriculum_state,
)
from mad_driving.training.metadata import (
    RESEARCH_CONTRACT_VERSION,
    checkpoint_curriculum_artifact_inventory,
    curriculum_state_artifact,
)
from mad_driving.training.train import TrainingResult, run_training


class FakeEnv(gym.Env[np.ndarray, int]):
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
    action_space = spaces.Discrete(4)

    def __init__(
        self,
        identifier: int,
        *,
        config: AppConfig,
        role: EnvironmentRole,
        worker_index: int,
    ) -> None:
        self.identifier = identifier
        self.config = config
        self.role = role
        self.worker_index = worker_index
        self.closed = False
        self.close_calls = 0
        self.reset_calls = 0
        self.difficulty_levels: list[int] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        super().reset(seed=seed)
        del options
        role_offset = 0 if self.role == "train" else 10_000
        episode_seed = role_offset + self.worker_index * 100 + self.reset_calls
        self.reset_calls += 1
        return np.zeros(24, dtype=np.float32), {
            "environment_seed": episode_seed,
            "scenario_selection_seed": episode_seed + 20_000,
            "scenario_parameter_seed": episode_seed + 30_000,
            "scenario_id": "nominal",
            "difficulty_level": 0,
            "scenario_parameters": {},
        }

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        return np.zeros(24, dtype=np.float32), 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1

    def set_difficulty_level(self, level: int) -> None:
        self.difficulty_levels.append(level)


@dataclass(frozen=True)
class EnvironmentCall:
    config: AppConfig
    role: EnvironmentRole
    worker_index: int


class EnvFactory:
    def __init__(self) -> None:
        self.created: list[FakeEnv] = []
        self.calls: list[EnvironmentCall] = []

    def __call__(
        self,
        config: AppConfig,
        *,
        role: EnvironmentRole,
        worker_index: int,
    ) -> FakeEnv:
        self.calls.append(EnvironmentCall(config, role, worker_index))
        env = FakeEnv(
            len(self.created),
            config=config,
            role=role,
            worker_index=worker_index,
        )
        self.created.append(env)
        return env


class CloseFailingEnv(FakeEnv):
    def close(self) -> None:
        super().close()
        raise OSError(f"environment {self.identifier} close failed")


class CloseFailingEnvFactory(EnvFactory):
    def __call__(
        self,
        config: AppConfig,
        *,
        role: EnvironmentRole,
        worker_index: int,
    ) -> FakeEnv:
        self.calls.append(EnvironmentCall(config, role, worker_index))
        env = CloseFailingEnv(
            len(self.created),
            config=config,
            role=role,
            worker_index=worker_index,
        )
        self.created.append(env)
        return env


class FakeVecEnv:
    def __init__(self, env_fns: list[Callable[[], FakeEnv]]) -> None:
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name: str) -> list[object]:
        return [env.get_wrapper_attr(attr_name) for env in self.envs]

    def env_method(self, method_name: str, *args: object) -> list[object]:
        return [getattr(env.unwrapped, method_name)(*args) for env in self.envs]


class VecFactory:
    def __init__(self) -> None:
        self.created: list[FakeVecEnv] = []

    def __call__(self, env_fns: list[Callable[[], FakeEnv]]) -> FakeVecEnv:
        vec_env = FakeVecEnv(env_fns)
        self.created.append(vec_env)
        return vec_env


class CallbackFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.instances: list[object] = []

    def __call__(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        callback = object()
        self.instances.append(callback)
        return callback


class FakeCheckpointCallback:
    def __init__(self, **kwargs: Any) -> None:
        self.controller = kwargs["controller"]
        self.save_freq = kwargs["save_freq"]
        self.save_path = Path(kwargs["save_path"])
        self.name_prefix = kwargs["name_prefix"]

    def on_fake_training(self, model: "FakePPO", produced_timesteps: int) -> None:
        train_env = model.init_kwargs["env"]
        start_timesteps = model.num_timesteps - produced_timesteps
        first_deadline = (start_timesteps // self.save_freq + 1) * self.save_freq
        checkpoint_timesteps_values = {
            math.ceil(deadline / train_env.num_envs) * train_env.num_envs
            for deadline in range(first_deadline, model.num_timesteps + 1, self.save_freq)
        }
        for checkpoint_timesteps in sorted(checkpoint_timesteps_values):
            if checkpoint_timesteps > model.num_timesteps:
                continue
            checkpoint = self.save_path / f"{self.name_prefix}_{checkpoint_timesteps}_steps.zip"
            model.write_checkpoint(checkpoint, "periodic")
            write_checkpoint_curriculum_state(self.controller.state, checkpoint)


class CheckpointCallbackFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.instances: list[FakeCheckpointCallback] = []

    def __call__(self, **kwargs: Any) -> FakeCheckpointCallback:
        self.calls.append(kwargs)
        callback = FakeCheckpointCallback(**kwargs)
        self.instances.append(callback)
        return callback


class FakeEvalCallback:
    def __init__(self, **kwargs: Any) -> None:
        self.controller = kwargs["controller"]
        self.eval_env = kwargs["eval_env"]
        self.eval_freq = kwargs["eval_freq"]
        self.best_model_save_path = Path(kwargs["best_model_save_path"])
        self.validation_episode_seed = kwargs["validation_episode_seed"]
        self.training_env_closed_during_evaluation: list[bool] = []

    def on_fake_training(self, model: "FakePPO", produced_timesteps: int) -> None:
        train_env = model.init_kwargs["env"]
        self.training_env_closed_during_evaluation.append(train_env.closed)
        for environment in self.eval_env.envs:
            environment.reset(seed=self.validation_episode_seed)
        start_timesteps = model.num_timesteps - produced_timesteps
        next_deadline = (start_timesteps // self.eval_freq + 1) * self.eval_freq
        if next_deadline <= model.num_timesteps:
            checkpoint = self.best_model_save_path / "best_model.zip"
            model.write_checkpoint(checkpoint, "best")
            write_checkpoint_curriculum_state(self.controller.state, checkpoint)


class EvalCallbackFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.instances: list[FakeEvalCallback] = []

    def __call__(self, **kwargs: Any) -> FakeEvalCallback:
        self.calls.append(kwargs)
        callback = FakeEvalCallback(**kwargs)
        self.instances.append(callback)
        return callback


class NoBestEvalCallback:
    def on_fake_training(self, model: "FakePPO", produced_timesteps: int) -> None:
        del model, produced_timesteps


class NoBestEvalCallbackFactory:
    def __call__(self, **kwargs: Any) -> NoBestEvalCallback:
        del kwargs
        return NoBestEvalCallback()


class RewardCallbackFactory:
    def __init__(self) -> None:
        self.instances: list[object] = []

    def __call__(self) -> object:
        callback = object()
        self.instances.append(callback)
        return callback


class FakePPO:
    instances: ClassVar[list["FakePPO"]] = []
    load_calls: ClassVar[list[dict[str, Any]]] = []
    fail_learn: ClassVar[bool] = False
    fail_save: ClassVar[bool] = False
    fail_init: ClassVar[bool] = False
    fail_load: ClassVar[bool] = False
    skip_save: ClassVar[bool] = False
    rollout_overshoot: ClassVar[int] = 0
    resume_num_timesteps: ClassVar[object] = 12_500

    @staticmethod
    def default_contract() -> dict[str, Any]:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "observation_shape": [24],
            "action_count": 4,
        }

    def __init__(self, policy: str, env: FakeVecEnv, **kwargs: Any) -> None:
        if type(self).fail_init:
            raise RuntimeError("PPO construction failed")
        self.init_kwargs = {"policy": policy, "env": env, **kwargs}
        self.learn_kwargs: dict[str, Any] | None = None
        self.saved_path: Path | None = None
        self.num_timesteps = 0
        self._set_contract({"policy": policy, **kwargs})
        type(self).instances.append(self)

    def _set_contract(self, values: dict[str, Any]) -> None:
        contract = {**self.default_contract(), **values}
        policy_name = str(contract["policy"])
        self.policy_class = type(
            "ActorCriticPolicy" if policy_name == "MlpPolicy" else policy_name,
            (),
            {},
        )
        self.learning_rate = contract["learning_rate"]
        if isinstance(self.learning_rate, int | float) and not isinstance(self.learning_rate, bool):
            self.lr_schedule = FloatSchedule(float(self.learning_rate))
        else:
            self.lr_schedule = self.learning_rate
        self.n_steps = contract["n_steps"]
        self.batch_size = contract["batch_size"]
        self.n_epochs = contract["n_epochs"]
        self.gamma = contract["gamma"]
        self.gae_lambda = contract["gae_lambda"]
        clip_range = contract["clip_range"]
        self.clip_range = (
            FloatSchedule(float(clip_range))
            if isinstance(clip_range, int | float) and not isinstance(clip_range, bool)
            else clip_range
        )
        self.ent_coef = contract["ent_coef"]
        self.vf_coef = contract["vf_coef"]
        self.max_grad_norm = contract["max_grad_norm"]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=tuple(contract["observation_shape"]),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(contract["action_count"])

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.load_calls = []
        cls.fail_learn = False
        cls.fail_save = False
        cls.fail_init = False
        cls.fail_load = False
        cls.skip_save = False
        cls.rollout_overshoot = 0
        cls.resume_num_timesteps = 12_500

    @classmethod
    def load(
        cls,
        path: object,
        env: FakeVecEnv | None = None,
        device: str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> "FakePPO":
        if hasattr(path, "read") and hasattr(path, "seek"):
            stream = path
            stream.seek(0)  # type: ignore[attr-defined]
            source_bytes = stream.read()  # type: ignore[attr-defined]
            stream.seek(0)  # type: ignore[attr-defined]
            checkpoint_source: object = io.BytesIO(source_bytes)
        else:
            source_bytes = Path(path).read_bytes()  # type: ignore[arg-type]
            checkpoint_source = path
        cls.load_calls.append(
            {
                "path": path,
                "source_bytes": source_bytes,
                "env": env,
                "device": device,
                "custom_objects": custom_objects,
                "print_system_info": print_system_info,
                "force_reset": force_reset,
                **kwargs,
            }
        )
        if cls.fail_load:
            raise RuntimeError("PPO load failed")
        model = cls.__new__(cls)
        model.init_kwargs = {"env": env}
        model.learn_kwargs = None
        model.saved_path = None
        model.num_timesteps = cls.resume_num_timesteps
        contract = cls.default_contract()
        if zipfile.is_zipfile(checkpoint_source):
            with zipfile.ZipFile(checkpoint_source) as checkpoint:
                if "ppo_contract.json" in checkpoint.namelist():
                    contract.update(json.loads(checkpoint.read("ppo_contract.json")))
        model._set_contract(contract)
        cls.instances.append(model)
        return model

    def learn(self, **kwargs: Any) -> "FakePPO":
        self.learn_kwargs = kwargs
        if type(self).fail_learn:
            raise RuntimeError("learn failed")
        produced_timesteps = kwargs["total_timesteps"] + type(self).rollout_overshoot
        self.num_timesteps += produced_timesteps
        for callback in kwargs["callback"]:
            on_fake_training = getattr(callback, "on_fake_training", None)
            if callable(on_fake_training):
                on_fake_training(self, produced_timesteps)
        return self

    def save(self, path: Path) -> None:
        self.saved_path = Path(path)
        if type(self).fail_save:
            raise RuntimeError("save failed")
        if not type(self).skip_save:
            self.write_checkpoint(self.saved_path, "final")

    @staticmethod
    def write_checkpoint(
        path: Path,
        marker: str,
        ppo_contract: dict[str, Any] | None = None,
    ) -> None:
        with zipfile.ZipFile(path, "w") as checkpoint:
            checkpoint.writestr("marker.txt", marker)
            if ppo_contract is not None:
                checkpoint.writestr(
                    "ppo_contract.json",
                    json.dumps(ppo_contract, sort_keys=True),
                )


class ResettingFakePPO(FakePPO):
    """Exercise initial and auto-reset-like calls at the real VecEnv boundary."""

    def learn(self, **kwargs: Any) -> FakePPO:
        train_env = self.init_kwargs["env"]
        for environment in train_env.envs:
            environment.reset()
            environment.reset()
        return super().learn(**kwargs)


@pytest.fixture(autouse=True)
def reset_fake_ppo() -> None:
    FakePPO.reset()


def make_config(*, num_envs: int = 1, **training_overrides: Any) -> AppConfig:
    training = {
        "num_envs": num_envs,
        "n_steps": 2048,
        "batch_size": 64,
        **training_overrides,
    }
    return AppConfig.model_validate(
        {
            "seed": 7,
            "scenario_id": "training-test",
            "decision_steps": 3,
            "fixed_action": [0.0, 0.0],
            "metadrive": {"use_render": False},
            "training": training,
        }
    )


def make_automatic_config() -> AppConfig:
    payload = make_config().model_dump(mode="python")
    payload["scenarios"]["selection"] = "auto"
    payload["scenarios"]["curriculum"] = {
        "mode": "automatic",
        "initial_level": 0,
        "success_rate_threshold": 0.8,
        "collision_rate_threshold": 0.05,
        "consecutive_evaluations": 2,
    }
    return AppConfig.model_validate(payload)


@dataclass(frozen=True)
class CompatibleSourceRun:
    run_dir: Path
    checkpoint: Path


def seed_compatible_source_run(
    tmp_path: Path,
    *,
    config: AppConfig | None = None,
    curriculum_state: CurriculumState | None = None,
    **ppo_overrides: Any,
) -> CompatibleSourceRun:
    index = 0
    while (tmp_path / f"source-run-{index}").exists():
        index += 1
    run_dir = tmp_path / f"source-run-{index}"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    selected_config = config or make_config()
    resolved_config = selected_config.model_dump(mode="json")
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    selected_state = curriculum_state or CurriculumState(
        level=0,
        consecutive_passes=0,
        evaluations=0,
    )
    curriculum_path = run_dir / "curriculum_state.yaml"
    write_curriculum_state(selected_state, curriculum_path)
    checkpoint = checkpoints_dir / "final_model.zip"
    FakePPO.write_checkpoint(
        checkpoint,
        "resume-source",
        {**FakePPO.default_contract(), **ppo_overrides},
    )
    write_checkpoint_curriculum_state(selected_state, checkpoint)
    metadata = RunMetadata(
        resolved_config=resolved_config,
        curriculum_state=curriculum_state_artifact(curriculum_path, selected_state),
        checkpoint_curriculum_artifacts=checkpoint_curriculum_artifact_inventory(checkpoints_dir),
    )
    metadata_module.write_run_metadata(metadata, run_dir / "run_metadata.json")
    return CompatibleSourceRun(run_dir=run_dir, checkpoint=checkpoint)


def add_periodic_checkpoint(
    source: CompatibleSourceRun,
    *,
    state: CurriculumState | None = None,
) -> Path:
    """Publish a second valid source checkpoint and refresh its metadata inventory."""

    selected_state = state or CurriculumState(level=0, consecutive_passes=0, evaluations=0)
    checkpoint = source.run_dir / "checkpoints" / "ppo_checkpoint_8_steps.zip"
    FakePPO.write_checkpoint(checkpoint, "periodic-resume-source")
    write_checkpoint_curriculum_state(selected_state, checkpoint)
    metadata_path = source.run_dir / "run_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["checkpoint_curriculum_artifacts"] = [
        dict(summary) for summary in checkpoint_curriculum_artifact_inventory(checkpoint.parent)
    ]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return checkpoint


def run_with_fakes(
    config: AppConfig,
    run_dir: Path,
    *,
    smoke: bool = True,
    resume_from: Path | None = None,
    env_factory: EnvFactory | None = None,
    dummy_factory: VecFactory | None = None,
    subproc_factory: Any = None,
    checkpoint_factory: CheckpointCallbackFactory | CallbackFactory | None = None,
    eval_factory: EvalCallbackFactory | CallbackFactory | None = None,
    reward_factory: RewardCallbackFactory | None = None,
    ppo_factory: type[FakePPO] = FakePPO,
) -> tuple[
    TrainingResult,
    EnvFactory,
    VecFactory,
    CheckpointCallbackFactory | CallbackFactory,
    EvalCallbackFactory | CallbackFactory,
]:
    environments = env_factory or EnvFactory()
    dummy = dummy_factory or VecFactory()
    checkpoint = checkpoint_factory or CheckpointCallbackFactory()
    evaluation = eval_factory or EvalCallbackFactory()
    reward = reward_factory or RewardCallbackFactory()
    subproc = subproc_factory or VecFactory()
    result = run_training(
        config,
        smoke=smoke,
        run_dir=run_dir,
        resume_from=resume_from,
        env_factory=environments,
        ppo_factory=ppo_factory,
        dummy_vec_env_factory=dummy,
        subproc_vec_env_factory=subproc,
        checkpoint_callback_factory=checkpoint,
        eval_callback_factory=evaluation,
        reward_callback_factory=reward,
    )
    return result, environments, dummy, checkpoint, evaluation


def test_nonempty_run_directory_is_rejected_before_any_side_effect(tmp_path: Path) -> None:
    run_dir = tmp_path / "occupied"
    run_dir.mkdir()
    marker = run_dir / "keep.txt"
    marker.write_bytes(b"original\x00bytes")
    environments = EnvFactory()

    with pytest.raises(FileExistsError) as raised:
        run_with_fakes(make_config(), run_dir, env_factory=environments)

    assert str(raised.value) == f"Run directory is non-empty: {run_dir}"
    assert marker.read_bytes() == b"original\x00bytes"
    assert list(run_dir.iterdir()) == [marker]
    assert environments.calls == []
    assert FakePPO.instances == []


def test_implicit_run_directory_race_is_rejected_without_deleting_competitor(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "implicit-race"
    run_dir.mkdir()
    environments = EnvFactory()

    with pytest.raises(FileExistsError, match="already exists|ownership"):
        run_training(
            make_config(),
            smoke=True,
            run_dir=run_dir,
            require_absent_run_dir=True,
            env_factory=environments,
            ppo_factory=FakePPO,
            dummy_vec_env_factory=VecFactory(),
            subproc_vec_env_factory=VecFactory(),
            checkpoint_callback_factory=CheckpointCallbackFactory(),
            eval_callback_factory=EvalCallbackFactory(),
            reward_callback_factory=RewardCallbackFactory(),
        )

    assert run_dir.is_dir()
    assert list(run_dir.iterdir()) == []
    assert environments.calls == []
    assert FakePPO.instances == []


def test_vector_constructor_error_keeps_primary_and_notes_owner_close_failure(
    tmp_path: Path,
) -> None:
    environments = CloseFailingEnvFactory()

    def fail_after_first_environment(
        env_fns: list[Callable[[], gym.Env[Any, Any]]],
    ) -> object:
        env_fns[0]()
        raise RuntimeError("vector construction failed")

    with pytest.raises(RuntimeError, match="vector construction failed") as captured:
        run_training(
            make_config(),
            smoke=True,
            run_dir=tmp_path / "constructor-close-failure",
            env_factory=environments,
            ppo_factory=FakePPO,
            dummy_vec_env_factory=fail_after_first_environment,
            subproc_vec_env_factory=VecFactory(),
            checkpoint_callback_factory=CheckpointCallbackFactory(),
            eval_callback_factory=EvalCallbackFactory(),
            reward_callback_factory=RewardCallbackFactory(),
        )

    assert environments.created[0].close_calls == 1
    assert any("environment 0 close failed" in note for note in captured.value.__notes__)


def test_environment_close_failure_prevents_success_publication(tmp_path: Path) -> None:
    run_dir = tmp_path / "close-failure"
    environments = CloseFailingEnvFactory()

    with pytest.raises(OSError) as captured:
        run_with_fakes(
            make_config(),
            run_dir,
            env_factory=environments,
        )

    assert not run_dir.exists()
    assert [environment.close_calls for environment in environments.created] == [1, 1]
    reported_errors = [str(captured.value), *captured.value.__notes__]
    assert any("environment 0 close failed" in message for message in reported_errors)
    assert any("environment 1 close failed" in message for message in reported_errors)


def test_seed_artifact_finalization_failure_prevents_success_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "seed-finalization-failure"
    environments = EnvFactory()

    def fail_finalization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("seed artifact finalization failed")

    monkeypatch.setattr(
        train_module,
        "summarize_episode_seed_artifacts",
        fail_finalization,
    )

    with pytest.raises(RuntimeError, match="seed artifact finalization failed"):
        run_with_fakes(
            make_config(),
            run_dir,
            env_factory=environments,
        )

    assert not run_dir.exists()
    assert all(environment.closed for environment in environments.created)


def test_logger_close_failure_is_not_retried_and_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "logger-close-failure"
    close_calls = 0

    def fail_logger_close(model: object | None) -> None:
        nonlocal close_calls
        assert model is not None
        close_calls += 1
        raise OSError("logger close failed")

    monkeypatch.setattr(train_module, "_close_model_logger", fail_logger_close)

    with pytest.raises(OSError, match="logger close failed"):
        run_with_fakes(make_config(), run_dir)

    assert close_calls == 1
    assert not run_dir.exists()


def test_existing_empty_run_directory_is_accepted(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()

    result, *_ = run_with_fakes(make_config(), run_dir)

    assert result.run_dir == run_dir
    assert (run_dir / "run_metadata.json").is_file()


def test_success_retires_claim_as_explicit_recovery_without_marker_in_final_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "published"

    run_with_fakes(make_config(), run_dir)

    recoveries = list(tmp_path.glob(".published.ownership-recovery-*"))
    assert len(recoveries) == 1
    assert recoveries[0].is_dir()
    marker = recoveries[0] / ".training-owner"
    assert marker.is_file()
    assert len(marker.read_bytes()) == 64
    assert not (run_dir / ".training-owner").exists()


def test_absent_destination_occupied_during_atomic_claim_preserves_foreign_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "raced-absent"
    foreign = run_dir / "competitor.bin"
    environments = EnvFactory()
    real_rename = ownership_module._rename_no_replace
    injected = False

    def claim_then_compete(source: Path, destination: Path) -> None:
        nonlocal injected
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            destination_path == run_dir
            and ".ownership-recovery-" in source_path.name
            and not injected
        ):
            injected = True
            run_dir.mkdir()
            foreign.write_bytes(b"competitor-owned\x00")
        real_rename(source, destination)

    monkeypatch.setattr(ownership_module, "_rename_no_replace", claim_then_compete)

    with pytest.raises(FileExistsError) as raised:
        run_with_fakes(make_config(), run_dir, env_factory=environments)

    assert str(raised.value) == f"Run directory is non-empty: {run_dir}"
    assert foreign.read_bytes() == b"competitor-owned\x00"
    assert list(run_dir.iterdir()) == [foreign]
    assert environments.calls == []
    assert FakePPO.instances == []


def test_empty_destination_occupied_during_exclusive_marker_preserves_foreign_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "raced-empty"
    run_dir.mkdir()
    foreign = run_dir / "competitor.bin"
    environments = EnvFactory()
    real_rename = ownership_module._rename_no_replace
    injected = False

    def move_then_compete(source: Path, destination: Path) -> None:
        nonlocal injected
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == run_dir and ".training-" in destination_path.name and not injected:
            injected = True
            foreign.write_bytes(b"competitor-owned\x00")
        real_rename(source, destination)

    monkeypatch.setattr(ownership_module, "_rename_no_replace", move_then_compete)

    with pytest.raises(FileExistsError) as raised:
        run_with_fakes(make_config(), run_dir, env_factory=environments)

    assert str(raised.value) == f"Run directory is non-empty: {run_dir}"
    assert foreign.read_bytes() == b"competitor-owned\x00"
    assert list(run_dir.iterdir()) == [foreign]
    assert environments.calls == []
    assert FakePPO.instances == []


def test_concurrent_compliant_trainers_have_one_owner_before_environment_side_effects(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "concurrent"
    first_reached_environment = threading.Event()
    release_first = threading.Event()
    first_environments = EnvFactory()
    second_environments = EnvFactory()
    first_errors: list[BaseException] = []

    class BlockingFactory:
        def __call__(
            self,
            config: AppConfig,
            *,
            role: EnvironmentRole,
            worker_index: int,
        ) -> FakeEnv:
            first_reached_environment.set()
            if not release_first.wait(timeout=5.0):
                raise TimeoutError("test did not release first trainer")
            return first_environments(config, role=role, worker_index=worker_index)

    def train_first() -> None:
        try:
            run_with_fakes(make_config(), run_dir, env_factory=BlockingFactory())  # type: ignore[arg-type]
        except BaseException as exc:
            first_errors.append(exc)

    thread = threading.Thread(target=train_first)
    thread.start()
    assert first_reached_environment.wait(timeout=5.0)
    marker_present = (run_dir / ".training-owner").is_file()
    try:
        with pytest.raises(FileExistsError):
            run_with_fakes(make_config(), run_dir, env_factory=second_environments)
    finally:
        release_first.set()
        thread.join(timeout=10.0)

    assert thread.is_alive() is False
    assert marker_present is True
    assert first_errors == []
    assert second_environments.calls == []
    assert not (run_dir / ".training-owner").exists()


@pytest.mark.parametrize("preexisting", [False, True])
def test_failed_marker_initialization_preserves_primary_error_and_cleans_owned_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: bool,
) -> None:
    run_dir = tmp_path / "marker-write-failure"
    if preexisting:
        run_dir.mkdir()
    environments = EnvFactory()

    def fail_marker_write(descriptor: int, data: bytes) -> int:
        del descriptor, data
        raise OSError("marker token write failed")

    monkeypatch.setattr(ownership_module.os, "write", fail_marker_write)

    with pytest.raises(OSError, match="marker token write failed"):
        run_with_fakes(make_config(), run_dir, env_factory=environments)

    assert environments.calls == []
    assert FakePPO.instances == []
    if preexisting:
        assert run_dir.is_dir()
        assert list(run_dir.iterdir()) == []
    else:
        assert not run_dir.exists()


def test_foreign_entry_added_after_acquisition_blocks_all_final_artifact_publication(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "occupied-after-acquire"
    foreign = run_dir / "foreign.bin"

    class OccupyingFactory(EnvFactory):
        def __call__(
            self,
            config: AppConfig,
            *,
            role: EnvironmentRole,
            worker_index: int,
        ) -> FakeEnv:
            if not foreign.exists():
                foreign.write_bytes(b"foreign-after-acquire\x00")
            return super().__call__(config, role=role, worker_index=worker_index)

    environments = OccupyingFactory()

    with pytest.raises(FileExistsError, match="ownership|publication|non-empty"):
        run_with_fakes(make_config(), run_dir, env_factory=environments)

    assert foreign.read_bytes() == b"foreign-after-acquire\x00"
    assert list(run_dir.iterdir()) == [foreign]
    assert not (run_dir / "config_resolved.yaml").exists()
    assert not (run_dir / "run_metadata.json").exists()
    assert not (run_dir / "checkpoints").exists()


@pytest.mark.parametrize("foreign_kind", ["file-entry", "empty-directory"])
def test_foreign_destination_created_at_atomic_publish_boundary_wins_without_coexistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    foreign_kind: str,
) -> None:
    run_dir = tmp_path / "occupied-at-publish"
    foreign = run_dir / "foreign.bin"
    real_rename = ownership_module._rename_no_replace
    injected = False
    foreign_directory_stat: os.stat_result | None = None

    def occupy_before_publish(source: Path, destination: Path) -> None:
        nonlocal foreign_directory_stat, injected
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == run_dir and ".training-" in source_path.name and not injected:
            injected = True
            run_dir.mkdir()
            foreign_directory_stat = run_dir.stat()
            if foreign_kind == "file-entry":
                foreign.write_bytes(b"foreign-at-publication\x00")
        real_rename(source, destination)

    monkeypatch.setattr(ownership_module, "_rename_no_replace", occupy_before_publish)

    with pytest.raises(FileExistsError, match="publication|ownership|non-empty"):
        run_with_fakes(make_config(), run_dir)

    assert injected is True
    assert foreign_directory_stat is not None
    assert os.path.samestat(foreign_directory_stat, run_dir.stat())
    if foreign_kind == "file-entry":
        assert foreign.read_bytes() == b"foreign-at-publication\x00"
        assert list(run_dir.iterdir()) == [foreign]
    else:
        assert list(run_dir.iterdir()) == []


def test_marker_replacement_during_failed_run_release_is_never_unlinked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "marker-replaced-at-release"
    marker = run_dir / ".training-owner"
    real_rename = ownership_module._rename_no_replace
    injected = False
    FakePPO.fail_learn = True

    def replace_marker_before_retirement(source: Path, destination: Path) -> None:
        nonlocal injected
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path == run_dir
            and ".ownership-recovery-" in destination_path.name
            and not injected
        ):
            injected = True
            marker.unlink()
            marker.write_bytes(b"foreign-replacement\x00")
        real_rename(source, destination)

    monkeypatch.setattr(ownership_module, "_rename_no_replace", replace_marker_before_retirement)

    with pytest.raises(RuntimeError, match="learn failed"):
        run_with_fakes(make_config(), run_dir)

    assert injected is True
    assert marker.read_bytes() == b"foreign-replacement\x00"
    assert list(run_dir.iterdir()) == [marker]


@pytest.mark.parametrize("case", ["malformed-source", "inside-source"])
def test_resume_read_only_preflight_performs_no_destination_write_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    if case == "malformed-source":
        (source.run_dir / "run_metadata.json").write_text("{malformed", encoding="utf-8")
        destination = tmp_path / "never-touched"
        expected = "metadata"
    else:
        destination = source.run_dir / "continued"
        expected = "separate"
    source_before = source_tree_bytes(source.run_dir)
    write_attempts: list[str] = []
    real_mkdir = Path.mkdir
    real_open = ownership_module.os.open

    def record_mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
        write_attempts.append(f"mkdir:{path}")
        real_mkdir(path, *args, **kwargs)

    def record_open(path: str | bytes | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if flags & (os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_RDWR):
            write_attempts.append(f"open:{path}")
        return real_open(path, flags, mode)

    monkeypatch.setattr(Path, "mkdir", record_mkdir)
    monkeypatch.setattr(ownership_module.os, "open", record_open)

    with pytest.raises(ValueError, match=expected):
        run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert write_attempts == []
    assert source_tree_bytes(source.run_dir) == source_before
    assert not destination.exists()


def test_run_directory_file_is_rejected_and_preserved_before_any_side_effect(
    tmp_path: Path,
) -> None:
    run_file = tmp_path / "occupied.bin"
    run_file.write_bytes(b"do-not-overwrite\x00")
    environments = EnvFactory()

    with pytest.raises(NotADirectoryError) as raised:
        run_with_fakes(make_config(), run_file, env_factory=environments)

    assert str(raised.value) == f"Run directory is not a directory: {run_file}"
    assert run_file.read_bytes() == b"do-not-overwrite\x00"
    assert environments.calls == []
    assert FakePPO.instances == []


def test_single_environment_uses_dummy_train_and_subprocess_validation_during_learning(
    tmp_path: Path,
) -> None:
    config = make_config(
        learning_rate=0.0007,
        n_epochs=4,
        gamma=0.97,
        gae_lambda=0.91,
        clip_range=0.15,
        ent_coef=0.02,
        vf_coef=0.4,
        max_grad_norm=0.7,
        seed=123,
    )
    run_dir = tmp_path / "run"

    subproc = VecFactory()
    result, environments, dummy, _, evaluation = run_with_fakes(
        config,
        run_dir,
        subproc_factory=subproc,
    )

    model = FakePPO.instances[0]
    train_env = dummy.created[0]
    eval_env = evaluation.instances[0].eval_env
    tensorboard_log = Path(model.init_kwargs["tensorboard_log"])
    private_workspace = tensorboard_log.parent
    assert tensorboard_log.name == "tensorboard"
    assert private_workspace.parent == run_dir.parent
    assert private_workspace.name.startswith(f".{run_dir.name}.training-")
    assert not private_workspace.exists()
    assert (run_dir / "tensorboard").is_dir()
    assert model.init_kwargs == {
        "policy": "MlpPolicy",
        "env": train_env,
        "learning_rate": 0.0007,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 4,
        "gamma": 0.97,
        "gae_lambda": 0.91,
        "clip_range": 0.15,
        "ent_coef": 0.02,
        "vf_coef": 0.4,
        "max_grad_norm": 0.7,
        "seed": 123,
        "tensorboard_log": str(tensorboard_log),
        "device": "cpu",
    }
    assert model.learn_kwargs is not None
    assert model.learn_kwargs["total_timesteps"] == 5_000
    assert model.learn_kwargs["reset_num_timesteps"] is True
    assert train_env is not eval_env
    assert dummy.created == [train_env]
    assert subproc.created == [eval_env]
    assert evaluation.instances[0].validation_episode_seed == config.seed
    assert evaluation.instances[0].training_env_closed_during_evaluation == [False]
    assert evaluation.instances[0] in model.learn_kwargs["callback"]
    assert environments.created[0] is not environments.created[1]
    assert all(env.closed for env in environments.created)
    assert all(env.close_calls == 1 for env in environments.created)
    assert all(vec.closed for vec in [*dummy.created, *subproc.created])
    assert result == TrainingResult(
        run_dir=run_dir,
        final_checkpoint=run_dir / "checkpoints" / "final_model.zip",
        best_checkpoint=run_dir / "checkpoints" / "best_model.zip",
        timesteps=5_000,
    )
    assert result.final_checkpoint.is_file()
    assert result.best_checkpoint.is_file()
    assert zipfile.is_zipfile(result.final_checkpoint)
    assert zipfile.is_zipfile(result.best_checkpoint)
    assert "log_path" not in evaluation.calls[0]
    assert not (run_dir / "evaluation").exists()


def test_normal_training_uses_500_000_timesteps(tmp_path: Path) -> None:
    result, *_ = run_with_fakes(make_config(), tmp_path / "normal", smoke=False)

    assert FakePPO.instances[0].learn_kwargs is not None
    assert FakePPO.instances[0].learn_kwargs["total_timesteps"] == 500_000
    assert result.timesteps == 500_000


def test_reports_actual_rollout_transitions_when_ppo_overshoots(tmp_path: Path) -> None:
    FakePPO.rollout_overshoot = 1_144

    result, *_ = run_with_fakes(make_config(), tmp_path / "overshoot")

    assert FakePPO.instances[0].num_timesteps == 6_144
    assert result.timesteps == 6_144


def test_writes_exact_stably_ordered_resolved_yaml(tmp_path: Path) -> None:
    config = make_config()
    run_dir = tmp_path / "yaml"

    run_with_fakes(config, run_dir)

    expected = yaml.safe_dump(
        config.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
    )
    assert (run_dir / "config_resolved.yaml").read_text(encoding="utf-8") == expected


def test_fresh_run_writes_complete_research_contract_metadata(tmp_path: Path) -> None:
    config = make_config()
    run_dir = tmp_path / "metadata"

    run_with_fakes(config, run_dir)

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    summaries = metadata.pop("episode_seed_artifacts")
    curriculum = metadata.pop("curriculum_state")
    checkpoint_curriculum = metadata.pop("checkpoint_curriculum_artifacts")
    assert metadata == {
        "research_contract_version": RESEARCH_CONTRACT_VERSION,
        "observation_schema_version": 1,
        "observation_shape": [24],
        "observation_dtype": "float32",
        "action_schema_version": 1,
        "action_count": 4,
        "action_order": ["KEEP", "SLOW", "PREPARE_STOP", "STOP"],
        "method_profile": {
            "method_id": get_method_profile(config.method.id).method_id,
            "policy_kind": get_method_profile(config.method.id).policy_kind,
            "specialist_ids": list(get_method_profile(config.method.id).specialist_ids),
            "critic_enabled": get_method_profile(config.method.id).critic_enabled,
            "shield_mode": get_method_profile(config.method.id).default_shield_mode,
        },
        "resolved_config": config.model_dump(mode="json"),
        "resume": None,
    }
    assert {descriptor["checkpoint_path"] for descriptor in checkpoint_curriculum} == {
        "checkpoints/best_model.zip",
        "checkpoints/final_model.zip",
    }
    assert all(descriptor["level"] == 0 for descriptor in checkpoint_curriculum)
    assert all(descriptor["consecutive_passes"] == 0 for descriptor in checkpoint_curriculum)
    assert all(descriptor["evaluations"] == 0 for descriptor in checkpoint_curriculum)
    curriculum_path = run_dir / "curriculum_state.yaml"
    assert curriculum == {
        "path": "curriculum_state.yaml",
        "sha256": hashlib.sha256(curriculum_path.read_bytes()).hexdigest(),
        "level": 0,
        "consecutive_passes": 0,
        "evaluations": 0,
    }
    assert [
        (summary["path"], summary["record_count"], summary["role"], summary["worker_index"])
        for summary in summaries
    ] == [
        ("episode_seeds/train-worker-000.jsonl", 0, "train", 0),
        ("episode_seeds/validation-worker-000.jsonl", 1, "validation", 0),
    ]
    for summary in summaries:
        artifact = run_dir / summary["path"]
        stat_result = os.stat(artifact, follow_symlinks=False)
        assert summary["schema_version"] == 4
        assert summary["file_identity"] == {
            "device": stat_result.st_dev,
            "inode": stat_result.st_ino,
        }
        assert summary["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_training_persists_actual_reset_seed_artifacts_by_role_and_worker(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "actual-reset-seeds"

    _, environments, *_ = run_with_fakes(
        make_config(num_envs=2),
        run_dir,
        subproc_factory=VecFactory(),
        ppo_factory=ResettingFakePPO,
    )

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    summaries = metadata["episode_seed_artifacts"]
    assert [(item["role"], item["worker_index"], item["record_count"]) for item in summaries] == [
        ("train", 0, 2),
        ("train", 1, 2),
        ("validation", 0, 1),
    ]
    records = {
        item["path"]: [
            json.loads(line)
            for line in (run_dir / item["path"]).read_text(encoding="utf-8").splitlines()
            if "environment_seed" in line
        ]
        for item in summaries
    }
    train_worker_0 = records["episode_seeds/train-worker-000.jsonl"]
    train_worker_1 = records["episode_seeds/train-worker-001.jsonl"]
    assert [record["environment_seed"] for record in train_worker_0] == [0, 1]
    assert [record["environment_seed"] for record in train_worker_1] == [100, 101]
    assert records["episode_seeds/validation-worker-000.jsonl"][0]["environment_seed"] == 10_000
    assert all(
        set(record)
        == {
            "role",
            "worker_index",
            "environment_seed",
            "scenario_selection_seed",
            "scenario_parameter_seed",
            "scenario_id",
            "difficulty_level",
            "scenario_parameters",
        }
        for worker_records in records.values()
        for record in worker_records
    )
    assert all(environment.closed for environment in environments.created)
    assert not list(tmp_path.glob(".actual-reset-seeds.training-*"))


def test_seed_artifacts_are_closed_and_summarized_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environments = EnvFactory()
    original_publish = ownership_module.RunDirectoryOwnership.publish
    observations: list[tuple[bool, list[int]]] = []

    def observe_publish(ownership: ownership_module.RunDirectoryOwnership) -> None:
        metadata = json.loads(
            (ownership.workspace / "run_metadata.json").read_text(encoding="utf-8")
        )
        observations.append(
            (
                all(environment.closed for environment in environments.created),
                [item["record_count"] for item in metadata["episode_seed_artifacts"]],
            )
        )
        original_publish(ownership)

    monkeypatch.setattr(ownership_module.RunDirectoryOwnership, "publish", observe_publish)

    run_with_fakes(
        make_config(),
        tmp_path / "publish-order",
        env_factory=environments,
        ppo_factory=ResettingFakePPO,
    )

    assert observations == [(True, [2, 1])]


class DescriptorTrackingVecEnv(FakeVecEnv):
    def __init__(self, env_fns: list[Callable[[], FakeEnv]]) -> None:
        super().__init__(env_fns)
        self.events: list[str] = []

    def get_attr(self, attr_name: str) -> list[object]:
        self.events.append(f"get_attr:{attr_name}:closed={self.closed}")
        return super().get_attr(attr_name)

    def close(self) -> None:
        self.events.append(f"close:closed={self.closed}")
        super().close()


class DescriptorTrackingVecFactory:
    def __init__(self) -> None:
        self.created: list[DescriptorTrackingVecEnv] = []

    def __call__(self, env_fns: list[Callable[[], FakeEnv]]) -> DescriptorTrackingVecEnv:
        vector_env = DescriptorTrackingVecEnv(env_fns)
        self.created.append(vector_env)
        return vector_env


class DescriptorFailingVecEnv(FakeVecEnv):
    def get_attr(self, attr_name: str) -> list[object]:
        del attr_name
        raise EOFError("worker descriptor channel closed")


class DescriptorFailingVecFactory:
    def __call__(self, env_fns: list[Callable[[], FakeEnv]]) -> DescriptorFailingVecEnv:
        return DescriptorFailingVecEnv(env_fns)


class DescriptorResponseVecEnv(FakeVecEnv):
    def __init__(self, env_fns: list[Callable[[], FakeEnv]], *, mode: str) -> None:
        super().__init__(env_fns)
        self.mode = mode

    def get_attr(self, attr_name: str) -> Any:
        descriptors = super().get_attr(attr_name)
        if self.mode == "nonsequence":
            return object()
        if self.mode == "incomplete":
            return descriptors[:-1]
        if self.mode == "malformed":
            return [object(), *descriptors[1:]]
        if self.mode == "mismatched":
            return [descriptors[0]._replace(worker_index=99), *descriptors[1:]]
        if self.mode == "duplicate":
            return [descriptors[0], descriptors[0]]
        raise AssertionError(f"unknown descriptor response mode: {self.mode}")


class DescriptorResponseVecFactory:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __call__(self, env_fns: list[Callable[[], FakeEnv]]) -> DescriptorResponseVecEnv:
        return DescriptorResponseVecEnv(env_fns, mode=self.mode)


def test_training_passes_parent_held_open_writer_descriptors_to_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_vectors = DescriptorTrackingVecFactory()
    validation_vectors = DescriptorTrackingVecFactory()
    received_descriptors: list[object] = []

    def capture_inventory(
        workspace: Path,
        *,
        expected_descriptors: tuple[object, ...],
    ) -> tuple[dict[str, object], ...]:
        del workspace
        received_descriptors.extend(expected_descriptors)
        return ()

    monkeypatch.setattr(
        train_module,
        "summarize_episode_seed_artifacts",
        capture_inventory,
    )

    run_with_fakes(
        make_config(num_envs=2),
        tmp_path / "parent-held-descriptors",
        dummy_factory=validation_vectors,
        subproc_factory=train_vectors,
    )

    assert [descriptor.role for descriptor in received_descriptors] == [
        "train",
        "train",
        "validation",
    ]
    assert [descriptor.worker_index for descriptor in received_descriptors] == [0, 1, 0]
    for vector_env in (*train_vectors.created, *validation_vectors.created):
        assert vector_env.events[0] == ("get_attr:episode_seed_artifact_descriptor:closed=False")
        assert vector_env.events[1] == "close:closed=False"


def test_descriptor_collection_failure_blocks_inventory_and_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "descriptor-channel-failed"
    inventory_called = False

    def unexpected_inventory(*args: object, **kwargs: object) -> tuple[object, ...]:
        del args, kwargs
        nonlocal inventory_called
        inventory_called = True
        return ()

    monkeypatch.setattr(
        train_module,
        "summarize_episode_seed_artifacts",
        unexpected_inventory,
    )

    with pytest.raises(RuntimeError, match="failed to report"):
        run_with_fakes(
            make_config(),
            run_dir,
            dummy_factory=DescriptorFailingVecFactory(),  # type: ignore[arg-type]
        )

    assert inventory_called is False
    assert not run_dir.exists()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("nonsequence", "malformed seed artifact identities"),
        ("incomplete", "incomplete seed artifact inventory"),
        ("malformed", "malformed seed artifact descriptor"),
        ("mismatched", "mismatched seed artifact descriptor"),
        ("duplicate", "duplicate seed artifact descriptors"),
    ],
)
def test_malformed_parent_descriptor_collection_blocks_publication(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    run_dir = tmp_path / f"bad-parent-descriptor-{mode}"

    with pytest.raises(RuntimeError, match=message):
        run_with_fakes(
            make_config(num_envs=2),
            run_dir,
            subproc_factory=DescriptorResponseVecFactory(mode),
        )

    assert not run_dir.exists()


def test_callback_frequencies_are_absolute_model_timesteps(tmp_path: Path) -> None:
    checkpoint = CallbackFactory()
    evaluation = EvalCallbackFactory()
    config = make_config(
        num_envs=4,
        checkpoint_interval_steps=10_003,
        eval_interval_steps=8_003,
    )

    run_with_fakes(
        config,
        tmp_path / "scaled",
        smoke=False,
        subproc_factory=VecFactory(),
        checkpoint_factory=checkpoint,
        eval_factory=evaluation,
    )

    assert checkpoint.calls[0]["save_freq"] == 10_003
    assert evaluation.calls[0]["eval_freq"] == 8_003


def test_callback_frequencies_have_a_minimum_of_one(tmp_path: Path) -> None:
    checkpoint = CallbackFactory()
    evaluation = EvalCallbackFactory()
    config = make_config(
        num_envs=4,
        checkpoint_interval_steps=1,
        eval_interval_steps=1,
    )

    run_with_fakes(
        config,
        tmp_path / "minimum",
        subproc_factory=VecFactory(),
        checkpoint_factory=checkpoint,
        eval_factory=evaluation,
    )

    assert checkpoint.calls[0]["save_freq"] == 1
    assert evaluation.calls[0]["eval_freq"] == 1


def test_training_wires_initial_curriculum_controller_and_owned_state_path(
    tmp_path: Path,
) -> None:
    evaluation = EvalCallbackFactory()
    run_dir = tmp_path / "curriculum-callback-wiring"

    run_with_fakes(make_config(), run_dir, eval_factory=evaluation)

    call = evaluation.calls[0]
    controller = call["controller"]
    assert controller.state == CurriculumState(0, 0, 0)
    assert Path(call["curriculum_state_path"]).name == "curriculum_state.yaml"


def test_failed_best_validation_does_not_publish_a_best_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "missing-best"
    best_checkpoint = run_dir / "checkpoints" / "best_model.zip"
    environments = EnvFactory()

    with pytest.raises(FileNotFoundError, match="Best checkpoint"):
        run_with_fakes(
            make_config(),
            run_dir,
            env_factory=environments,
            eval_factory=NoBestEvalCallbackFactory(),
        )

    assert not best_checkpoint.exists()
    assert all(env.close_calls == 1 for env in environments.created)


def test_scheduled_evaluation_publishes_best_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "fresh-best"
    best_checkpoint = run_dir / "checkpoints" / "best_model.zip"

    result, *_ = run_with_fakes(make_config(), run_dir)

    assert result.best_checkpoint == best_checkpoint
    assert zipfile.is_zipfile(best_checkpoint)
    with zipfile.ZipFile(best_checkpoint) as checkpoint:
        assert checkpoint.read("marker.txt") == b"best"


def test_failed_final_validation_does_not_publish_a_final_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "missing-final"
    final_checkpoint = run_dir / "checkpoints" / "final_model.zip"
    FakePPO.skip_save = True
    environments = EnvFactory()

    with pytest.raises(FileNotFoundError, match="Final checkpoint"):
        run_with_fakes(make_config(), run_dir, env_factory=environments)

    assert not final_checkpoint.exists()
    assert all(env.close_calls == 1 for env in environments.created)


def assert_no_invocation_staging(run_dir: Path) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return
    assert [path for path in checkpoints_dir.iterdir() if path.name.startswith(".training-")] == []


class FailingEnvFactory:
    def __call__(
        self,
        config: AppConfig,
        *,
        role: EnvironmentRole,
        worker_index: int,
    ) -> FakeEnv:
        del config, role, worker_index
        raise RuntimeError("environment construction failed")


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("environment", "environment construction failed"),
        ("ppo", "PPO load failed"),
        ("learn", "learn failed"),
        ("save", "save failed"),
        ("best_validation", "Best checkpoint"),
        ("final_validation", "Final checkpoint"),
    ],
)
def test_failed_resume_preserves_all_source_run_artifacts(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    source_before = source_tree_bytes(source.run_dir)
    run_dir = tmp_path / failure
    env_factory: Any = EnvFactory()
    eval_factory: Any = EvalCallbackFactory()
    if failure == "environment":
        env_factory = FailingEnvFactory()
    elif failure == "ppo":
        FakePPO.fail_load = True
    elif failure == "learn":
        FakePPO.fail_learn = True
    elif failure == "save":
        FakePPO.fail_save = True
    elif failure == "best_validation":
        eval_factory = NoBestEvalCallbackFactory()
    elif failure == "final_validation":
        FakePPO.skip_save = True

    with pytest.raises((RuntimeError, FileNotFoundError), match=message):
        run_with_fakes(
            make_config(),
            run_dir,
            resume_from=source.checkpoint,
            env_factory=env_factory,
            dummy_factory=VecFactory(),
            eval_factory=eval_factory,
        )

    assert source_tree_bytes(source.run_dir) == source_before
    if (run_dir / "checkpoints").exists():
        assert_no_invocation_staging(run_dir)


def test_staging_cleanup_failure_does_not_mask_training_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakePPO.fail_learn = True

    def fail_staging_cleanup(path: Path) -> None:
        del path
        raise OSError("staging cleanup failed")

    monkeypatch.setattr(train_module.shutil, "rmtree", fail_staging_cleanup)

    with pytest.raises(RuntimeError, match="learn failed") as raised:
        run_with_fakes(make_config(), tmp_path / "learn-and-staging-cleanup")

    assert any("staging cleanup failed" in note for note in raised.value.__notes__)


def test_staging_cleanup_failure_replaces_success_with_explicit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_staging_cleanup(path: Path) -> None:
        del path
        raise OSError("staging cleanup failed")

    monkeypatch.setattr(train_module.shutil, "rmtree", fail_staging_cleanup)

    with pytest.raises(RuntimeError, match="Training resource cleanup failed"):
        run_with_fakes(make_config(), tmp_path / "successful-staging-cleanup-failure")


def test_success_stages_then_promotes_best_final_and_periodic_checkpoints(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "transaction"
    checkpoint = CheckpointCallbackFactory()
    evaluation = EvalCallbackFactory()

    result, *_ = run_with_fakes(
        make_config(checkpoint_interval_steps=2_500),
        run_dir,
        checkpoint_factory=checkpoint,
        eval_factory=evaluation,
    )

    staging_dir = Path(checkpoint.calls[0]["save_path"])
    assert staging_dir == Path(evaluation.calls[0]["best_model_save_path"])
    private_workspace = staging_dir.parent.parent
    assert private_workspace.parent == run_dir.parent
    assert private_workspace.name.startswith(f".{run_dir.name}.training-")
    assert staging_dir.name.startswith(".training-")
    assert FakePPO.instances[0].saved_path == staging_dir / "final_model.zip"
    assert not staging_dir.exists()
    periodic_checkpoints = sorted((run_dir / "checkpoints").glob("ppo_checkpoint_*_steps.zip"))
    assert result.final_checkpoint == run_dir / "checkpoints" / "final_model.zip"
    assert result.best_checkpoint == run_dir / "checkpoints" / "best_model.zip"
    assert [checkpoint.name for checkpoint in periodic_checkpoints] == [
        "ppo_checkpoint_2500_steps.zip",
        "ppo_checkpoint_5000_steps.zip",
    ]
    for periodic_checkpoint in periodic_checkpoints:
        assert zipfile.is_zipfile(periodic_checkpoint)
        with zipfile.ZipFile(periodic_checkpoint) as checkpoint_file:
            assert checkpoint_file.read("marker.txt") == b"periodic"


@pytest.mark.parametrize("failure_at", [2, 3])
def test_promotion_failure_removes_every_new_canonical_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: int,
) -> None:
    run_dir = tmp_path / f"replace-{failure_at}"
    checkpoints_dir = run_dir / "checkpoints"
    real_replace = train_module.os.replace
    promotion_calls = 0

    def fail_selected_replace(source: Path, destination: Path) -> None:
        nonlocal promotion_calls
        if Path(source).parent.name.startswith(".training-"):
            promotion_calls += 1
        if promotion_calls == failure_at:
            raise OSError(f"replacement {failure_at} failed")
        real_replace(source, destination)

    monkeypatch.setattr(train_module.os, "replace", fail_selected_replace)

    with pytest.raises(OSError, match=f"replacement {failure_at} failed"):
        run_with_fakes(
            make_config(checkpoint_interval_steps=2_500),
            run_dir,
        )

    assert not list(checkpoints_dir.glob("*.zip"))
    assert_no_invocation_staging(run_dir)


def test_resume_destination_inside_source_run_is_rejected_without_source_changes(
    tmp_path: Path,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    source_before = source_tree_bytes(source.run_dir)
    destination = source.run_dir / "continued"

    with pytest.raises(ValueError, match="separate"):
        run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert source_tree_bytes(source.run_dir) == source_before
    assert not destination.exists()
    assert FakePPO.load_calls == []


def test_num_envs_above_one_uses_subprocess_train_env_and_separate_dummy_eval(
    tmp_path: Path,
) -> None:
    subproc = VecFactory()
    dummy = VecFactory()
    environments = EnvFactory()

    run_with_fakes(
        make_config(num_envs=3),
        tmp_path / "subproc",
        env_factory=environments,
        dummy_factory=dummy,
        subproc_factory=subproc,
    )

    assert len(subproc.created) == 1
    assert subproc.created[0].num_envs == 3
    assert len(dummy.created) == 1
    assert dummy.created[0].num_envs == 1
    assert len({env.identifier for env in environments.created}) == 4
    assert all(env.closed for env in environments.created)
    assert all(env.close_calls == 1 for env in environments.created)


def test_training_constructs_train_workers_and_validation_eval(tmp_path: Path) -> None:
    environments = EnvFactory()

    run_with_fakes(
        make_config(num_envs=3),
        tmp_path / "role-aware",
        env_factory=environments,
        subproc_factory=VecFactory(),
    )

    assert [(call.role, call.worker_index) for call in environments.calls] == [
        ("train", 0),
        ("train", 1),
        ("train", 2),
        ("validation", 0),
    ]
    assert all(call.role != "test" for call in environments.calls)


def test_environment_thunks_capture_independent_config_copies(tmp_path: Path) -> None:
    config = make_config(num_envs=3)
    environments = EnvFactory()

    run_with_fakes(
        config,
        tmp_path / "independent-configs",
        env_factory=environments,
        subproc_factory=VecFactory(),
    )

    captured_configs = [call.config for call in environments.calls]
    assert len(captured_configs) == 4
    assert all(captured is not config for captured in captured_configs)
    assert len({id(captured) for captured in captured_configs}) == 4
    assert all(captured == config for captured in captured_configs)


@pytest.mark.parametrize(
    ("role", "worker_index", "message"),
    [
        ("invalid", 0, "role"),
        ("train", -1, "worker_index"),
        ("train", True, "worker_index"),
    ],
)
def test_default_environment_factory_rejects_invalid_identity(
    role: str,
    worker_index: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        train_module._default_env_factory(  # type: ignore[arg-type]
            make_config(),
            role=role,
            worker_index=worker_index,
        )


class FailingSubprocVecEnv:
    latest: ClassVar["FailingSubprocVecEnv | None"] = None

    def __init__(self, env_fns: list[Callable[[], FakeEnv]]) -> None:
        self.envs = [env_fns[0]()]
        self.closed = False
        type(self).latest = self
        raise OSError("Windows spawn failed")

    def close(self) -> None:
        self.closed = True
        for env in self.envs:
            env.close()


def test_subprocess_failure_closes_partial_resources_and_fails_without_dummy_fallback(
    tmp_path: Path,
) -> None:
    dummy = VecFactory()
    environments = EnvFactory()

    with pytest.raises(OSError, match="Windows spawn failed"):
        run_with_fakes(
            make_config(num_envs=3),
            tmp_path / "fallback",
            env_factory=environments,
            dummy_factory=dummy,
            subproc_factory=FailingSubprocVecEnv,
        )

    assert FailingSubprocVecEnv.latest is not None
    assert FailingSubprocVecEnv.latest.closed is True
    assert FailingSubprocVecEnv.latest.envs[0].unwrapped.closed is True
    assert FailingSubprocVecEnv.latest.envs[0].unwrapped.close_calls == 1
    assert dummy.created == []
    assert len(environments.created) == 1
    assert all(env.closed for env in environments.created)
    assert all(env.close_calls == 1 for env in environments.created)


class FakeRemote:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class FailingCloseRemote(FakeRemote):
    def close(self) -> None:
        super().close()
        raise OSError("remote close failed")


class FakeProcess:
    def __init__(
        self,
        *,
        exits_on_terminate: bool,
        exits_on_kill: bool,
        initially_alive: bool = True,
    ) -> None:
        self.alive = initially_alive
        self.exitcode: int | None = None if initially_alive else 0
        self.exits_on_terminate = exits_on_terminate
        self.exits_on_kill = exits_on_kill
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_calls: list[float | None] = []
        self.is_alive_calls = 0

    def is_alive(self) -> bool:
        self.is_alive_calls += 1
        return self.alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.exits_on_terminate:
            self.alive = False
            self.exitcode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        if self.exits_on_kill:
            self.alive = False
            self.exitcode = -9

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


class OperationFailingProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__(exits_on_terminate=False, exits_on_kill=True)
        self.join_failures_remaining = 1

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        if self.join_failures_remaining:
            self.join_failures_remaining -= 1
            raise OSError("join failed")

    def terminate(self) -> None:
        self.terminate_calls += 1
        raise OSError("terminate failed")


class DeadlineProcess(FakeProcess):
    def __init__(
        self,
        clock: list[float],
        *,
        exits_after_join: bool,
    ) -> None:
        super().__init__(exits_on_terminate=True, exits_on_kill=True)
        self.clock = clock
        self.exits_after_join = exits_after_join

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        assert timeout is not None
        self.clock[0] += timeout
        if self.exits_after_join:
            self.alive = False
            self.exitcode = 0


class PostTerminateJoinProcess(FakeProcess):
    def __init__(self, clock: list[float]) -> None:
        super().__init__(exits_on_terminate=False, exits_on_kill=True)
        self.clock = clock
        self.terminated = False

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.terminated = True

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        assert timeout is not None
        self.clock[0] += timeout
        if self.terminated and timeout > 0:
            self.alive = False
            self.exitcode = -15


class PlannedJoinProcess(FakeProcess):
    def __init__(
        self,
        clock: list[float],
        join_plan: list[tuple[bool, bool]],
    ) -> None:
        super().__init__(exits_on_terminate=False, exits_on_kill=False)
        self.clock = clock
        self.join_plan = list(join_plan)

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        assert timeout is not None
        consume_timeout, exit_after_join = self.join_plan.pop(0)
        if consume_timeout:
            self.clock[0] += timeout
        if exit_after_join:
            self.alive = False
            self.exitcode = -9 if self.kill_calls else (-15 if self.terminate_calls else 0)


def test_graceful_shutdown_redistributes_fast_first_worker_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [10.0]
    fast = PlannedJoinProcess(clock, [(False, True)])
    slow = PlannedJoinProcess(clock, [(True, True)])
    monkeypatch.setattr(train_module, "_monotonic", lambda: clock[0], raising=False)

    workers_alive, escalated, operation_errors = train_module._stop_processes(
        [fast, slow],
        graceful_join=True,
    )

    assert fast.join_calls == [pytest.approx(1.5)]
    assert slow.join_calls == [pytest.approx(3.0)]
    assert clock[0] == pytest.approx(13.0)
    assert workers_alive is False
    assert escalated is False
    assert operation_errors == ()


def test_terminate_shutdown_redistributes_fast_first_worker_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [20.0]
    fast = PlannedJoinProcess(clock, [(False, True)])
    slow = PlannedJoinProcess(clock, [(True, True)])
    monkeypatch.setattr(train_module, "_monotonic", lambda: clock[0], raising=False)

    workers_alive, escalated, operation_errors = train_module._stop_processes(
        [fast, slow],
        graceful_join=False,
    )

    assert fast.join_calls == [pytest.approx(0.5)]
    assert slow.join_calls == [pytest.approx(1.0)]
    assert clock[0] == pytest.approx(21.0)
    assert all(process.terminate_calls == 1 for process in (fast, slow))
    assert all(process.kill_calls == 0 for process in (fast, slow))
    assert workers_alive is False
    assert escalated is True
    assert operation_errors == ()


def test_kill_shutdown_redistributes_fast_first_worker_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [30.0]
    fast = PlannedJoinProcess(clock, [(True, False), (False, True)])
    slow = PlannedJoinProcess(clock, [(True, False), (True, True)])
    monkeypatch.setattr(train_module, "_monotonic", lambda: clock[0], raising=False)

    workers_alive, escalated, operation_errors = train_module._stop_processes(
        [fast, slow],
        graceful_join=False,
    )

    assert fast.join_calls == [pytest.approx(0.5), pytest.approx(0.5)]
    assert slow.join_calls == [pytest.approx(0.5), pytest.approx(1.0)]
    assert clock[0] == pytest.approx(32.0)
    assert all(process.terminate_calls == 1 for process in (fast, slow))
    assert all(process.kill_calls == 1 for process in (fast, slow))
    assert workers_alive is False
    assert escalated is True
    assert operation_errors == ()


def test_multiworker_shutdown_uses_one_shared_five_second_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    first = DeadlineProcess(clock, exits_after_join=True)
    second = DeadlineProcess(clock, exits_after_join=False)
    monkeypatch.setattr(
        train_module,
        "_monotonic",
        lambda: clock[0],
        raising=False,
    )

    workers_alive, escalated, operation_errors = train_module._stop_processes(
        [first, second],
        graceful_join=True,
    )

    assert first.join_calls == [1.5]
    assert second.join_calls == [1.5, 1.0]
    assert first.terminate_calls == 0
    assert second.terminate_calls == 1
    assert clock[0] == 104.0
    assert workers_alive is False
    assert escalated is True
    assert operation_errors == ()


def test_multiworker_shutdown_reserves_positive_fair_post_terminate_reaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [200.0]
    processes = [PostTerminateJoinProcess(clock), PostTerminateJoinProcess(clock)]
    monkeypatch.setattr(train_module, "_monotonic", lambda: clock[0], raising=False)

    workers_alive, escalated, operation_errors = train_module._stop_processes(
        processes,
        graceful_join=True,
    )

    assert workers_alive is False
    assert escalated is True
    assert operation_errors == ()
    assert all(process.terminate_calls == 1 for process in processes)
    assert all(process.kill_calls == 0 for process in processes)
    assert all(len(process.join_calls) == 2 for process in processes)
    assert all(
        process.join_calls[1] is not None and process.join_calls[1] > 0 for process in processes
    )
    assert (
        sum(
            timeout
            for process in processes
            for timeout in process.join_calls
            if timeout is not None
        )
        <= 5.0
    )


def test_failed_worker_shutdown_does_not_mark_close_audit_complete() -> None:
    vector_env = RuntimeProcessVecEnv(
        [
            lambda: FakeEnv(
                0,
                config=make_config(),
                role="train",
                worker_index=0,
            )
        ],
        exits_on_terminate=False,
        exits_on_kill=False,
    )
    vector_env.remotes = (FakeRemote(),)

    with pytest.raises(RuntimeError, match="cleanup could not be confirmed"):
        train_module._close_vector_env(vector_env)

    assert not getattr(vector_env, train_module._VECTOR_CLOSE_AUDIT_MARKER, False)


class ProcessFailingSubprocVecEnv:
    latest: ClassVar["ProcessFailingSubprocVecEnv | None"] = None

    def __init__(self, env_fns: list[Callable[[], FakeEnv]]) -> None:
        del env_fns
        self.remotes = (FakeRemote(), FakeRemote())
        self.work_remotes = (FakeRemote(), FakeRemote())
        self.processes = [
            FakeProcess(exits_on_terminate=True, exits_on_kill=True),
            FakeProcess(exits_on_terminate=False, exits_on_kill=True),
            FakeProcess(
                exits_on_terminate=True,
                exits_on_kill=True,
                initially_alive=False,
            ),
        ]
        type(self).latest = self
        raise OSError("subprocess handshake failed")


class UnsafeProcessFailingSubprocVecEnv:
    latest: ClassVar["UnsafeProcessFailingSubprocVecEnv | None"] = None

    def __init__(self, env_fns: list[Callable[[], FakeEnv]]) -> None:
        del env_fns
        self.remotes = (FakeRemote(),)
        self.work_remotes = (FakeRemote(),)
        self.processes = [FakeProcess(exits_on_terminate=False, exits_on_kill=False)]
        type(self).latest = self
        raise OSError("subprocess handshake failed")


class OperationFailingSubprocVecEnv:
    latest: ClassVar["OperationFailingSubprocVecEnv | None"] = None

    def __init__(self, env_fns: list[Callable[[], FakeEnv]]) -> None:
        del env_fns
        self.remotes = (FakeRemote(),)
        self.work_remotes = (FakeRemote(),)
        self.processes = [OperationFailingProcess()]
        type(self).latest = self
        raise OSError("subprocess operation failed")


class RemoteCloseFailingSubprocVecEnv:
    def __init__(self, env_fns: list[Callable[[], FakeEnv]]) -> None:
        del env_fns
        self.remotes = (FailingCloseRemote(),)
        self.work_remotes = (FakeRemote(),)
        self.processes = [
            FakeProcess(
                exits_on_terminate=True,
                exits_on_kill=True,
                initially_alive=False,
            )
        ]
        raise OSError("remote constructor failed")


def test_subprocess_cleanup_confirms_workers_dead_before_reporting_construction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_module, "_monotonic", lambda: 0.0, raising=False)
    dummy = VecFactory()

    with pytest.raises(OSError, match="subprocess handshake failed"):
        run_with_fakes(
            make_config(num_envs=3),
            tmp_path / "process-fallback",
            dummy_factory=dummy,
            subproc_factory=ProcessFailingSubprocVecEnv,
        )

    partial = ProcessFailingSubprocVecEnv.latest
    assert partial is not None
    assert all(remote.close_calls == 1 for remote in (*partial.remotes, *partial.work_remotes))
    terminated, killed, already_dead = partial.processes
    assert terminated.terminate_calls == 1
    assert terminated.kill_calls == 0
    assert terminated.join_calls == [pytest.approx(0.5)]
    assert killed.terminate_calls == 1
    assert killed.kill_calls == 1
    assert killed.join_calls == [pytest.approx(1.0), pytest.approx(1.0)]
    assert already_dead.terminate_calls == 0
    assert already_dead.kill_calls == 0
    assert already_dead.join_calls == []
    assert all(process.is_alive() is False for process in partial.processes)
    assert all(process.is_alive_calls >= 2 for process in partial.processes)
    assert dummy.created == []


def test_subprocess_cleanup_refuses_fallback_when_worker_survives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_module, "_monotonic", lambda: 0.0, raising=False)
    dummy = VecFactory()

    with pytest.raises(OSError, match="subprocess handshake failed") as raised:
        run_with_fakes(
            make_config(num_envs=3),
            tmp_path / "unsafe-process",
            dummy_factory=dummy,
            subproc_factory=UnsafeProcessFailingSubprocVecEnv,
        )

    partial = UnsafeProcessFailingSubprocVecEnv.latest
    assert partial is not None
    assert all(remote.close_calls == 1 for remote in (*partial.remotes, *partial.work_remotes))
    process = partial.processes[0]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.join_calls == [pytest.approx(1.0), pytest.approx(1.0)]
    assert process.is_alive() is True
    assert dummy.created == []
    assert any("worker cleanup could not be confirmed" in note for note in raised.value.__notes__)


def test_partial_cleanup_continues_to_kill_after_join_and_terminate_fail(
    tmp_path: Path,
) -> None:
    with pytest.raises(OSError, match="subprocess operation failed"):
        run_with_fakes(
            make_config(num_envs=3),
            tmp_path / "operation-failure",
            subproc_factory=OperationFailingSubprocVecEnv,
        )

    partial = OperationFailingSubprocVecEnv.latest
    assert partial is not None
    process = partial.processes[0]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.is_alive() is False


def test_partial_remote_close_failure_is_noted_without_masking_constructor(
    tmp_path: Path,
) -> None:
    with pytest.raises(OSError, match="remote constructor failed") as raised:
        run_with_fakes(
            make_config(num_envs=3),
            tmp_path / "remote-close-failure",
            subproc_factory=RemoteCloseFailingSubprocVecEnv,
        )

    assert any("remote cleanup could not be confirmed" in note for note in raised.value.__notes__)


class GracefulRemote(FakeRemote):
    def __init__(
        self,
        process: FakeProcess,
        environments: list[gym.Env[Any, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.process = process
        self.environments = environments or []
        self.sent: list[tuple[str, None]] = []

    def send(self, message: tuple[str, None]) -> None:
        self.sent.append(message)
        if message == ("close", None):
            for environment in self.environments:
                environment.close()
            self.process.alive = False
            self.process.exitcode = 0


class SilentWorkerCloseFailureRemote(GracefulRemote):
    def send(self, message: tuple[str, None]) -> None:
        self.sent.append(message)
        if message == ("close", None):
            for environment in self.environments:
                try:
                    environment.close()
                except OSError:
                    pass
            self.process.alive = False
            self.process.exitcode = 1


class ControlledExitCodeRemote(GracefulRemote):
    def __init__(self, process: FakeProcess, *, exitcode: object) -> None:
        super().__init__(process)
        self.controlled_exitcode = exitcode

    def send(self, message: tuple[str, None]) -> None:
        super().send(message)
        self.process.exitcode = self.controlled_exitcode  # type: ignore[assignment]


class UnconfirmedEscalatedProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__(exits_on_terminate=False, exits_on_kill=False)

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.alive = False


class DelayedGracefulProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__(exits_on_terminate=True, exits_on_kill=True)

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        if timeout is not None and timeout >= 2.0:
            self.alive = False
            self.exitcode = 0


class DelayedCloseRemote(FakeRemote):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple[str, None]] = []

    def send(self, message: tuple[str, None]) -> None:
        self.sent.append(message)


class RuntimeCloseFailingRemote(GracefulRemote):
    def close(self) -> None:
        super().close()
        raise OSError("runtime remote close failed")


class RuntimeProcessVecEnv(FakeVecEnv):
    def __init__(
        self,
        env_fns: list[Callable[[], FakeEnv]],
        *,
        exits_on_terminate: bool = True,
        exits_on_kill: bool = True,
    ) -> None:
        super().__init__(env_fns)
        self.processes = [
            FakeProcess(
                exits_on_terminate=exits_on_terminate,
                exits_on_kill=exits_on_kill,
            )
        ]
        self.remotes = (GracefulRemote(self.processes[0], self.envs),)
        self.waiting = False

    def close(self) -> None:
        raise AssertionError("process vector env must use bounded worker shutdown")


class RuntimeProcessVecFactory:
    def __init__(
        self,
        *,
        graceful: bool = True,
        exits_on_terminate: bool = True,
        exits_on_kill: bool = True,
    ) -> None:
        self.graceful = graceful
        self.exits_on_terminate = exits_on_terminate
        self.exits_on_kill = exits_on_kill
        self.created: list[RuntimeProcessVecEnv] = []

    def __call__(self, env_fns: list[Callable[[], FakeEnv]]) -> RuntimeProcessVecEnv:
        vector_env = RuntimeProcessVecEnv(
            env_fns,
            exits_on_terminate=self.exits_on_terminate,
            exits_on_kill=self.exits_on_kill,
        )
        if not self.graceful:
            vector_env.remotes = (FakeRemote(),)
        self.created.append(vector_env)
        return vector_env


class NonzeroExitRuntimeProcessVecFactory:
    def __init__(self) -> None:
        self.created: list[RuntimeProcessVecEnv] = []

    def __call__(self, env_fns: list[Callable[[], FakeEnv]]) -> RuntimeProcessVecEnv:
        vector_env = RuntimeProcessVecEnv(env_fns)
        vector_env.remotes = (
            SilentWorkerCloseFailureRemote(vector_env.processes[0], vector_env.envs),
        )
        self.created.append(vector_env)
        return vector_env


def test_successful_training_confirms_subprocess_training_worker_exited(
    tmp_path: Path,
) -> None:
    subproc = RuntimeProcessVecFactory()

    run_with_fakes(
        make_config(num_envs=2),
        tmp_path / "bounded-close",
        subproc_factory=subproc,
    )

    training = subproc.created[0]
    process = training.processes[0]
    remote = training.remotes[0]
    assert remote.sent == [("close", None)]
    assert remote.close_calls == 1
    assert process.join_calls == [pytest.approx(3.0)]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.is_alive() is False
    assert training.closed is True


def test_subprocess_nonzero_exit_without_pipe_error_is_a_cleanup_failure() -> None:
    vector_env = RuntimeProcessVecEnv(
        [
            lambda: CloseFailingEnv(
                0,
                config=make_config(),
                role="train",
                worker_index=0,
            )
        ]
    )
    process = vector_env.processes[0]
    remote = SilentWorkerCloseFailureRemote(process, vector_env.envs)
    vector_env.remotes = (remote,)

    with pytest.raises(RuntimeError, match="exit code 1"):
        train_module._close_vector_env(vector_env)

    assert remote.sent == [("close", None)]
    assert remote.close_calls == 1
    assert process.is_alive() is False
    assert process.exitcode == 1
    assert vector_env.closed is True


def test_preclosed_subprocess_still_requires_first_exitcode_audit() -> None:
    vector_env = RuntimeProcessVecEnv(
        [
            lambda: FakeEnv(
                0,
                config=make_config(),
                role="train",
                worker_index=0,
            )
        ]
    )
    process = vector_env.processes[0]
    process.alive = False
    process.exitcode = 1
    vector_env.closed = True

    with pytest.raises(RuntimeError, match="exit code 1"):
        train_module._close_vector_env(vector_env)

    assert process.join_calls == [pytest.approx(3.0)]


@pytest.mark.parametrize(
    ("exitcode", "message"),
    [
        (None, "exit code is unconfirmed"),
        ("zero", "malformed exit code"),
    ],
)
def test_subprocess_requires_a_confirmed_integer_exit_code(
    exitcode: object,
    message: str,
) -> None:
    vector_env = RuntimeProcessVecEnv(
        [
            lambda: FakeEnv(
                0,
                config=make_config(),
                role="train",
                worker_index=0,
            )
        ]
    )
    process = vector_env.processes[0]
    vector_env.remotes = (ControlledExitCodeRemote(process, exitcode=exitcode),)

    with pytest.raises(RuntimeError, match=message):
        train_module._close_vector_env(vector_env)

    assert process.is_alive() is False
    assert vector_env.closed is True


def test_subprocess_none_exitcode_after_escalation_is_unconfirmed() -> None:
    vector_env = RuntimeProcessVecEnv(
        [
            lambda: FakeEnv(
                0,
                config=make_config(),
                role="train",
                worker_index=0,
            )
        ]
    )
    process = UnconfirmedEscalatedProcess()
    vector_env.processes = [process]
    vector_env.remotes = (FakeRemote(),)

    with pytest.raises(RuntimeError, match="exit code is unconfirmed") as raised:
        train_module._close_vector_env(vector_env)

    assert "terminate/kill escalation" in str(raised.value)
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.is_alive() is False
    assert not getattr(vector_env, train_module._VECTOR_CLOSE_AUDIT_MARKER, False)


def test_nonzero_worker_exit_blocks_inventory_and_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "nonzero-worker-exit"
    subproc = NonzeroExitRuntimeProcessVecFactory()
    inventory_called = False

    def unexpected_inventory(*args: object, **kwargs: object) -> tuple[object, ...]:
        del args, kwargs
        nonlocal inventory_called
        inventory_called = True
        return ()

    monkeypatch.setattr(
        train_module,
        "summarize_episode_seed_artifacts",
        unexpected_inventory,
    )

    with pytest.raises(RuntimeError, match="exit code 1"):
        run_with_fakes(
            make_config(num_envs=2),
            run_dir,
            subproc_factory=subproc,
        )

    assert inventory_called is False
    assert not run_dir.exists()


def test_nonzero_worker_exit_is_noted_without_masking_training_error(
    tmp_path: Path,
) -> None:
    FakePPO.fail_learn = True
    subproc = NonzeroExitRuntimeProcessVecFactory()

    with pytest.raises(RuntimeError, match="learn failed") as raised:
        run_with_fakes(
            make_config(num_envs=2),
            tmp_path / "learn-plus-nonzero-exit",
            subproc_factory=subproc,
        )

    assert any("exit code 1" in note for note in raised.value.__notes__)


def test_runtime_cleanup_escalation_fails_once_after_stopping_worker() -> None:
    vector_env = RuntimeProcessVecEnv(
        [
            lambda: FakeEnv(
                0,
                config=make_config(),
                role="train",
                worker_index=0,
            )
        ]
    )
    process = OperationFailingProcess()
    vector_env.processes = [process]
    vector_env.remotes = (FakeRemote(),)

    with pytest.raises(RuntimeError, match="terminate/kill escalation"):
        train_module._close_vector_env(vector_env)

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.is_alive() is False
    operation_counts = (process.terminate_calls, process.kill_calls, len(process.join_calls))

    train_module._close_vector_env(vector_env)

    assert (
        process.terminate_calls,
        process.kill_calls,
        len(process.join_calls),
    ) == operation_counts


def test_runtime_cleanup_allows_slow_graceful_metadrive_worker_exit() -> None:
    vector_env = RuntimeProcessVecEnv(
        [
            lambda: FakeEnv(
                0,
                config=make_config(),
                role="train",
                worker_index=0,
            )
        ]
    )
    process = DelayedGracefulProcess()
    remote = DelayedCloseRemote()
    vector_env.processes = [process]
    vector_env.remotes = (remote,)

    train_module._close_vector_env(vector_env)

    assert remote.sent == [("close", None)]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.exitcode == 0


def test_runtime_subprocess_remote_close_failure_is_propagated() -> None:
    vector_env = RuntimeProcessVecEnv(
        [
            lambda: FakeEnv(
                0,
                config=make_config(),
                role="train",
                worker_index=0,
            )
        ]
    )
    remote = RuntimeCloseFailingRemote(vector_env.processes[0])
    vector_env.remotes = (remote,)

    with pytest.raises(RuntimeError, match="remote cleanup.*could not be confirmed"):
        train_module._close_vector_env(vector_env)

    assert remote.close_calls == 1
    assert vector_env.processes[0].is_alive() is False
    assert vector_env.closed is True


def test_worker_cleanup_failure_replaces_success_with_explicit_error(tmp_path: Path) -> None:
    subproc = RuntimeProcessVecFactory(
        graceful=False,
        exits_on_terminate=False,
        exits_on_kill=False,
    )

    with pytest.raises(RuntimeError, match="worker cleanup could not be confirmed"):
        run_with_fakes(
            make_config(num_envs=2),
            tmp_path / "unsafe-close",
            subproc_factory=subproc,
        )

    process = subproc.created[0].processes[0]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.is_alive() is True


def test_worker_cleanup_failure_does_not_mask_training_error(tmp_path: Path) -> None:
    FakePPO.fail_learn = True
    subproc = RuntimeProcessVecFactory(
        graceful=False,
        exits_on_terminate=False,
        exits_on_kill=False,
    )

    with pytest.raises(RuntimeError, match="learn failed") as raised:
        run_with_fakes(
            make_config(num_envs=2),
            tmp_path / "learn-and-close-failure",
            subproc_factory=subproc,
        )

    assert any("worker cleanup could not be confirmed" in note for note in raised.value.__notes__)


class SerializingVecFactory(VecFactory):
    def __init__(self) -> None:
        super().__init__()
        self.serialized_thunks = 0

    def __call__(self, env_fns: list[Callable[[], FakeEnv]]) -> FakeVecEnv:
        for env_fn in env_fns:
            assert cloudpickle.loads(cloudpickle.dumps(env_fn)) is not None
            self.serialized_thunks += 1
        return super().__call__(env_fns)


class RoundTripVecFactory(VecFactory):
    def __call__(self, env_fns: list[Callable[[], FakeEnv]]) -> FakeVecEnv:
        restored = [cloudpickle.loads(cloudpickle.dumps(env_fn)) for env_fn in env_fns]
        return super().__call__(restored)


def test_every_train_and_eval_env_thunk_is_cloudpickle_serializable(tmp_path: Path) -> None:
    subproc = SerializingVecFactory()
    dummy = SerializingVecFactory()

    run_with_fakes(
        make_config(num_envs=3),
        tmp_path / "cloudpickle",
        dummy_factory=dummy,
        subproc_factory=subproc,
    )

    assert subproc.serialized_thunks == 3
    assert dummy.serialized_thunks == 1


def test_cloudpickle_round_trip_preserves_each_role_and_worker_without_shared_config(
    tmp_path: Path,
) -> None:
    subproc = RoundTripVecFactory()
    dummy = RoundTripVecFactory()

    run_with_fakes(
        make_config(num_envs=3),
        tmp_path / "cloudpickle-identities",
        dummy_factory=dummy,
        subproc_factory=subproc,
    )

    train_envs = subproc.created[0].envs
    eval_envs = dummy.created[0].envs
    assert [(env.unwrapped.role, env.unwrapped.worker_index) for env in train_envs] == [
        ("train", 0),
        ("train", 1),
        ("train", 2),
    ]
    assert [(env.unwrapped.role, env.unwrapped.worker_index) for env in eval_envs] == [
        ("validation", 0)
    ]
    configs = [env.unwrapped.config for env in (*train_envs, *eval_envs)]
    assert len({id(config) for config in configs}) == 4


def source_tree_bytes(run_dir: Path) -> dict[str, bytes]:
    return {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }


def test_resume_records_provenance_preserves_source_and_keeps_timesteps(
    tmp_path: Path,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    source_before = source_tree_bytes(source.run_dir)
    destination = tmp_path / "continued"
    config = make_config()
    FakePPO.rollout_overshoot = 96

    result, *_ = run_with_fakes(config, destination, resume_from=source.checkpoint)

    metadata = json.loads((destination / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["resume"] == {
        "parent_checkpoint_path": str(source.checkpoint.resolve()),
        "parent_checkpoint_sha256": sha256_file(source.checkpoint),
        "parent_run_dir": str(source.run_dir.resolve()),
        "parent_config": config.model_dump(mode="json"),
        "config_diff": {},
        "start_num_timesteps": 12_500,
    }
    assert isinstance(FakePPO.load_calls[0]["path"], io.BytesIO)
    assert FakePPO.load_calls[0]["source_bytes"] == source.checkpoint.read_bytes()
    assert FakePPO.instances[0].learn_kwargs is not None
    assert FakePPO.instances[0].learn_kwargs["reset_num_timesteps"] is False
    assert FakePPO.instances[0].num_timesteps == 17_596
    assert result.timesteps == 5_096
    assert source_tree_bytes(source.run_dir) == source_before


def test_resume_restores_exact_automatic_curriculum_state_without_regression(
    tmp_path: Path,
) -> None:
    config = make_automatic_config()
    parent_state = CurriculumState(level=2, consecutive_passes=1, evaluations=7)
    source = seed_compatible_source_run(
        tmp_path,
        config=config,
        curriculum_state=parent_state,
    )
    destination = tmp_path / "curriculum-continued"

    result, environments, *_ = run_with_fakes(
        config,
        destination,
        resume_from=source.checkpoint,
    )

    restored = read_curriculum_state(result.run_dir / "curriculum_state.yaml")
    assert restored == parent_state
    metadata = json.loads((result.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["curriculum_state"] == {
        "path": "curriculum_state.yaml",
        "sha256": sha256_file(result.run_dir / "curriculum_state.yaml"),
        "level": 2,
        "consecutive_passes": 1,
        "evaluations": 7,
    }
    assert all(environment.difficulty_levels == [2] for environment in environments.created)


@pytest.mark.parametrize("checkpoint_kind", ["periodic", "final"])
@pytest.mark.parametrize(
    "corruption",
    ["missing", "null-metadata", "hash-mismatch", "malformed", "metadata-values"],
)
def test_resume_rejects_invalid_run_final_curriculum_artifact_before_destination(
    tmp_path: Path,
    checkpoint_kind: str,
    corruption: str,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    selected_checkpoint = (
        add_periodic_checkpoint(source) if checkpoint_kind == "periodic" else source.checkpoint
    )
    state_path = source.run_dir / "curriculum_state.yaml"
    metadata_path = source.run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if corruption == "missing":
        state_path.unlink()
    elif corruption == "null-metadata":
        metadata["curriculum_state"] = None
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif corruption == "hash-mismatch":
        state_path.write_text(
            "level: 0\nconsecutive_passes: 0\nevaluations: 1\n",
            encoding="utf-8",
        )
    elif corruption == "malformed":
        state_path.write_text("level: [", encoding="utf-8")
        metadata["curriculum_state"]["sha256"] = sha256_file(state_path)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        metadata["curriculum_state"]["evaluations"] = 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    destination = tmp_path / f"rejected-final-state-{checkpoint_kind}-{corruption}"
    environments = EnvFactory()

    with pytest.raises(ValueError, match="curriculum|Curriculum|Artifact"):
        run_with_fakes(
            make_config(),
            destination,
            resume_from=selected_checkpoint,
            env_factory=environments,
        )

    assert not destination.exists()
    assert environments.calls == []
    assert FakePPO.load_calls == []


@pytest.mark.parametrize("checkpoint_kind", ["periodic", "final"])
def test_resume_validates_run_final_curriculum_reachability_against_parent_config(
    tmp_path: Path,
    checkpoint_kind: str,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    selected_checkpoint = (
        add_periodic_checkpoint(source) if checkpoint_kind == "periodic" else source.checkpoint
    )
    unreachable = CurriculumState(level=1, consecutive_passes=0, evaluations=1)
    state_path = source.run_dir / "curriculum_state.yaml"
    write_curriculum_state(unreachable, state_path)
    metadata_path = source.run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["curriculum_state"].update(
        {
            "sha256": sha256_file(state_path),
            "level": unreachable.level,
            "consecutive_passes": unreachable.consecutive_passes,
            "evaluations": unreachable.evaluations,
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed curriculum.*level"):
        run_with_fakes(
            make_config(),
            tmp_path / f"unreachable-final-state-{checkpoint_kind}",
            resume_from=selected_checkpoint,
        )

    assert FakePPO.load_calls == []


@pytest.mark.parametrize(
    "corruption",
    ["missing", "hash-mismatch", "malformed", "level-out-of-range", "metadata-mismatch"],
)
def test_resume_rejects_invalid_checkpoint_curriculum_state_before_destination(
    tmp_path: Path,
    corruption: str,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    state_path = checkpoint_curriculum_sidecar_path(source.checkpoint)
    metadata_path = source.run_dir / "run_metadata.json"
    if corruption == "missing":
        state_path.unlink()
    elif corruption == "hash-mismatch":
        state_path.write_text(
            "schema_version: 1\n"
            f"checkpoint_sha256: {sha256_file(source.checkpoint)}\n"
            "level: 0\nconsecutive_passes: 0\nevaluations: 1\n",
            encoding="utf-8",
        )
    elif corruption == "malformed":
        state_path.write_text("level: [", encoding="utf-8")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["checkpoint_curriculum_artifacts"][0]["state_sha256"] = sha256_file(state_path)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif corruption == "level-out-of-range":
        state_path.write_text(
            "schema_version: 1\n"
            f"checkpoint_sha256: {sha256_file(source.checkpoint)}\n"
            "level: 4\nconsecutive_passes: 0\nevaluations: 0\n",
            encoding="utf-8",
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["checkpoint_curriculum_artifacts"][0].update(
            {"state_sha256": sha256_file(state_path), "level": 4}
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["checkpoint_curriculum_artifacts"][0]["evaluations"] = 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    destination = tmp_path / f"rejected-curriculum-{corruption}"
    environments = EnvFactory()

    with pytest.raises(ValueError, match="curriculum|Curriculum"):
        run_with_fakes(
            make_config(),
            destination,
            resume_from=source.checkpoint,
            env_factory=environments,
        )

    assert not destination.exists()
    assert environments.calls == []
    assert FakePPO.load_calls == []


def test_resume_rejects_incompatible_curriculum_configuration(
    tmp_path: Path,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    payload = make_config().model_dump(mode="python")
    payload["scenarios"]["curriculum"]["success_rate_threshold"] = 0.9
    incompatible = AppConfig.model_validate(payload)

    with pytest.raises(ValueError, match="scenarios.curriculum.success_rate_threshold"):
        run_with_fakes(
            incompatible,
            tmp_path / "incompatible-curriculum",
            resume_from=source.checkpoint,
        )


@pytest.mark.parametrize(
    "corruption",
    ["missing", "malformed", "legacy", "wrong-observation", "wrong-action-order"],
)
def test_resume_rejects_invalid_source_metadata_before_destination_or_environment(
    tmp_path: Path,
    corruption: str,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    metadata_path = source.run_dir / "run_metadata.json"
    if corruption == "missing":
        metadata_path.unlink()
    elif corruption == "malformed":
        metadata_path.write_text("{not-json", encoding="utf-8")
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if corruption == "legacy":
            metadata["research_contract_version"] = 1
        elif corruption == "wrong-observation":
            metadata["observation_schema_version"] = 99
        else:
            metadata["action_order"] = ["STOP", "SLOW", "PREPARE_STOP", "KEEP"]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    destination = tmp_path / "rejected"
    environments = EnvFactory()

    with pytest.raises(ValueError, match="metadata|contract|schema|action"):
        run_with_fakes(
            make_config(),
            destination,
            resume_from=source.checkpoint,
            env_factory=environments,
        )

    assert not destination.exists()
    assert environments.calls == []
    assert FakePPO.load_calls == []


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_resume_rejects_non_finite_source_metadata_without_mutating_source_or_destination(
    tmp_path: Path,
    constant: str,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    metadata_path = source.run_dir / "run_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["resolved_config"]["training"]["seed"] = {
        "NaN": math.nan,
        "Infinity": math.inf,
        "-Infinity": -math.inf,
    }[constant]
    metadata_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    source_before = source_tree_bytes(source.run_dir)
    destination = tmp_path / "non-finite-source"
    environments = EnvFactory()

    with pytest.raises(ValueError, match="finite"):
        run_with_fakes(
            make_config(),
            destination,
            resume_from=source.checkpoint,
            env_factory=environments,
        )

    assert not destination.exists()
    assert source_tree_bytes(source.run_dir) == source_before
    assert environments.calls == []
    assert FakePPO.load_calls == []


def test_resume_rejects_nested_invalid_source_config_type_without_mutation(
    tmp_path: Path,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    metadata_path = source.run_dir / "run_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["resolved_config"]["training"]["seed"] = {"nested": [7]}
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    source_before = source_tree_bytes(source.run_dir)
    destination = tmp_path / "invalid-nested-source"

    with pytest.raises(ValueError, match="resolved config|metadata"):
        run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert not destination.exists()
    assert source_tree_bytes(source.run_dir) == source_before
    assert FakePPO.load_calls == []


@pytest.mark.parametrize(
    ("field", "source_value"),
    [
        ("policy", "CnnPolicy"),
        ("learning_rate", 0.001),
        ("learning_rate", "0.0003"),
        ("n_steps", 1024),
        ("batch_size", 32),
        ("n_epochs", 3),
        ("gamma", 0.9),
        ("gae_lambda", 0.8),
        ("clip_range", 0.1),
        ("clip_range", "0.2"),
        ("ent_coef", 0.2),
        ("vf_coef", 0.7),
        ("max_grad_norm", 0.9),
    ],
)
def test_resume_rejects_loaded_ppo_hyperparameter_mismatch_before_writes_or_learning(
    tmp_path: Path,
    field: str,
    source_value: object,
) -> None:
    source = seed_compatible_source_run(tmp_path, **{field: source_value})
    destination = tmp_path / "incompatible"

    with pytest.raises(ValueError, match=field):
        run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert not destination.exists()
    assert FakePPO.instances[0].learn_kwargs is None


def test_resume_rejects_nonconstant_effective_learning_rate_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    destination = tmp_path / "scheduled-learning-rate"
    original_load = FakePPO.load.__func__

    def scheduled_load(cls: type[FakePPO], path: Path, **kwargs: Any) -> FakePPO:
        model = original_load(cls, path, **kwargs)
        model.lr_schedule = lambda progress: 0.0003 if progress == 1.0 else 0.0001
        return model

    monkeypatch.setattr(FakePPO, "load", classmethod(scheduled_load))

    with pytest.raises(ValueError, match="learning_rate"):
        run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert not destination.exists()
    assert FakePPO.instances[0].learn_kwargs is None


def test_resume_rejects_callable_that_bypasses_three_point_schedule_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    destination = tmp_path / "bypass-schedule"
    original_load = FakePPO.load.__func__

    def bypass_load(cls: type[FakePPO], path: Path, **kwargs: Any) -> FakePPO:
        model = original_load(cls, path, **kwargs)
        model.lr_schedule = lambda progress: 0.0001 if progress == 0.25 else 0.0003
        return model

    monkeypatch.setattr(FakePPO, "load", classmethod(bypass_load))

    with pytest.raises(ValueError, match="learning_rate"):
        run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert not destination.exists()
    assert FakePPO.instances[0].learn_kwargs is None


def test_resume_accepts_exact_pinned_sb3_constant_schedule_representation(
    tmp_path: Path,
) -> None:
    source = seed_compatible_source_run(tmp_path)

    run_with_fakes(make_config(), tmp_path / "canonical-schedule", resume_from=source.checkpoint)

    model = FakePPO.instances[0]
    assert isinstance(model.lr_schedule, FloatSchedule)
    assert isinstance(model.lr_schedule.value_schedule, ConstantSchedule)
    assert isinstance(model.clip_range, FloatSchedule)
    assert isinstance(model.clip_range.value_schedule, ConstantSchedule)


@pytest.mark.parametrize("raw_timesteps", [True, 12.5, -1])
def test_resume_rejects_invalid_raw_num_timesteps_before_writes_or_learning(
    tmp_path: Path,
    raw_timesteps: object,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    destination = tmp_path / "invalid-num-timesteps"
    FakePPO.resume_num_timesteps = raw_timesteps

    with pytest.raises(ValueError, match="num_timesteps"):
        run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert not destination.exists()
    assert FakePPO.instances[0].learn_kwargs is None


@pytest.mark.parametrize(
    ("override", "message"),
    [({"observation_shape": [25]}, "observation"), ({"action_count": 5}, "action")],
)
def test_resume_rejects_loaded_space_mismatch_before_writes_or_learning(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    source = seed_compatible_source_run(tmp_path, **override)
    destination = tmp_path / "space-mismatch"

    with pytest.raises(ValueError, match=message):
        run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert not destination.exists()
    assert FakePPO.instances[0].learn_kwargs is None


def test_resume_config_diff_contains_only_changed_allowed_fields_in_sorted_order(
    tmp_path: Path,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    config = make_config(
        seed=43,
        smoke_timesteps=6_000,
        total_timesteps=600_000,
        checkpoint_interval_steps=11_000,
        eval_interval_steps=12_000,
        run_root="continued-runs",
    )
    destination = tmp_path / "allowed-diff"

    run_with_fakes(config, destination, resume_from=source.checkpoint)

    resume = json.loads((destination / "run_metadata.json").read_text(encoding="utf-8"))["resume"]
    expected = {
        "training.checkpoint_interval_steps": {"parent": 10_000, "current": 11_000},
        "training.eval_interval_steps": {"parent": 10_000, "current": 12_000},
        "training.run_root": {"parent": "runs", "current": "continued-runs"},
        "training.seed": {"parent": 42, "current": 43},
        "training.smoke_timesteps": {"parent": 5_000, "current": 6_000},
        "training.total_timesteps": {"parent": 500_000, "current": 600_000},
    }
    assert resume["config_diff"] == expected
    assert list(resume["config_diff"]) == sorted(expected)


def test_resume_rejects_disallowed_config_difference_before_environment_or_destination(
    tmp_path: Path,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    payload = make_config().model_dump(mode="json")
    payload["reward"]["offroad"] = 101.0
    config = AppConfig.model_validate(payload)
    destination = tmp_path / "different-reward"
    environments = EnvFactory()

    with pytest.raises(ValueError, match="reward.offroad"):
        run_with_fakes(
            config,
            destination,
            resume_from=source.checkpoint,
            env_factory=environments,
        )

    assert not destination.exists()
    assert environments.calls == []
    assert FakePPO.load_calls == []


@pytest.mark.parametrize("corruption", ["missing", "malformed", "inconsistent"])
def test_resume_rejects_invalid_resolved_source_config_before_destination(
    tmp_path: Path,
    corruption: str,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    config_path = source.run_dir / "config_resolved.yaml"
    if corruption == "missing":
        config_path.unlink()
    elif corruption == "malformed":
        config_path.write_text("training: [", encoding="utf-8")
    else:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["training"]["seed"] = 99
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    destination = tmp_path / "invalid-config"

    with pytest.raises(ValueError, match="resolved config|metadata"):
        run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert not destination.exists()
    assert FakePPO.load_calls == []


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("metadrive", "use_render"), 0),
        (("shield", "imminent_ttc_s"), True),
        (("shield", "imminent_ttc_s"), 1),
    ],
)
def test_resume_rejects_type_changed_metadata_resolved_config_scalars(
    tmp_path: Path,
    path: tuple[str, str],
    replacement: object,
) -> None:
    config = make_config()
    source = seed_compatible_source_run(tmp_path, config=config)
    metadata_path = source.run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["resolved_config"][path[0]][path[1]] = replacement
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    destination = tmp_path / f"type-changed-{path[1]}-{type(replacement).__name__}"
    environments = EnvFactory()

    with pytest.raises(ValueError, match="does not match"):
        run_with_fakes(
            config,
            destination,
            resume_from=source.checkpoint,
            env_factory=environments,
        )

    assert not destination.exists()
    assert environments.calls == []
    assert FakePPO.load_calls == []


@pytest.mark.parametrize("duplicate_scope", ["root", "nested"])
def test_resume_rejects_duplicate_resolved_config_keys_before_destination_or_ppo_load(
    tmp_path: Path,
    duplicate_scope: str,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    config_path = source.run_dir / "config_resolved.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    if duplicate_scope == "root":
        config_text = config_text.replace("seed: 7\n", "seed: 7\nseed: 7\n", 1)
    else:
        config_text = config_text.replace(
            "  seed: 42\n",
            "  seed: 42\n  seed: 42\n",
            1,
        )
    config_path.write_text(config_text, encoding="utf-8")
    destination = tmp_path / f"duplicate-config-{duplicate_scope}"

    with pytest.raises(ValueError, match="duplicate"):
        run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert not destination.exists()
    assert FakePPO.load_calls == []


def test_resume_load_uses_snapshot_if_checkpoint_path_changes_after_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    destination = tmp_path / "changed-source"
    original_load = FakePPO.load.__func__

    authenticated_bytes = source.checkpoint.read_bytes()

    def changing_load(cls: type[FakePPO], path: object, **kwargs: Any) -> FakePPO:
        model = original_load(cls, path, **kwargs)
        source.checkpoint.write_bytes(b"changed-during-load")
        return model

    monkeypatch.setattr(FakePPO, "load", classmethod(changing_load))

    run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert destination.is_dir()
    assert source.checkpoint.read_bytes() == b"changed-during-load"
    assert FakePPO.load_calls[0]["source_bytes"] == authenticated_bytes


def test_resume_loads_authenticated_checkpoint_snapshot_across_swap_and_restore_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    authenticated_bytes = source.checkpoint.read_bytes()
    replacement = tmp_path / "replacement.zip"
    FakePPO.write_checkpoint(replacement, "replacement-with-valid-contract")
    replacement_bytes = replacement.read_bytes()
    original_load = FakePPO.load.__func__

    def swapping_load(cls: type[FakePPO], path: object, **kwargs: Any) -> FakePPO:
        source.checkpoint.write_bytes(replacement_bytes)
        try:
            return original_load(cls, path, **kwargs)
        finally:
            source.checkpoint.write_bytes(authenticated_bytes)

    monkeypatch.setattr(FakePPO, "load", classmethod(swapping_load))

    run_with_fakes(
        make_config(),
        tmp_path / "immutable-checkpoint-resume",
        resume_from=source.checkpoint,
    )

    assert FakePPO.load_calls[0]["source_bytes"] == authenticated_bytes
    assert FakePPO.load_calls[0]["source_bytes"] != replacement_bytes


def test_resume_loads_ppo_with_train_env_and_keeps_timesteps(tmp_path: Path) -> None:
    source = seed_compatible_source_run(tmp_path)
    source_bytes = source.checkpoint.read_bytes()

    FakePPO.rollout_overshoot = 96
    result, *_ = run_with_fakes(
        make_config(),
        tmp_path / "resume",
        resume_from=source.checkpoint,
    )

    assert len(FakePPO.load_calls) == 1
    assert len(FakePPO.instances) == 1
    model = FakePPO.instances[0]
    tensorboard_log = Path(FakePPO.load_calls[0]["tensorboard_log"])
    private_workspace = tensorboard_log.parent
    assert private_workspace.parent == tmp_path
    assert private_workspace.name.startswith(".resume.training-")
    assert not private_workspace.exists()
    assert (tmp_path / "resume" / "tensorboard").is_dir()
    load_call = dict(FakePPO.load_calls[0])
    assert isinstance(load_call.pop("path"), io.BytesIO)
    assert load_call == {
        "source_bytes": source_bytes,
        "env": model.init_kwargs["env"],
        "device": "cpu",
        "custom_objects": None,
        "print_system_info": False,
        "force_reset": True,
        "tensorboard_log": str(tensorboard_log),
    }
    assert model.learn_kwargs is not None
    assert model.learn_kwargs["reset_num_timesteps"] is False
    assert model.num_timesteps == 17_596
    assert result.timesteps == 5_096


def test_ppo_construction_failure_closes_each_environment_once(tmp_path: Path) -> None:
    FakePPO.fail_init = True
    environments = EnvFactory()

    with pytest.raises(RuntimeError, match="PPO construction failed"):
        run_with_fakes(
            make_config(num_envs=2),
            tmp_path / "init-failure",
            env_factory=environments,
            subproc_factory=VecFactory(),
        )

    assert environments.created
    assert all(env.close_calls == 1 for env in environments.created)


@pytest.mark.parametrize(
    ("failure", "message"),
    [("learn", "learn failed"), ("save", "save failed")],
)
def test_training_failure_still_closes_every_environment(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    environments = EnvFactory()
    if failure == "learn":
        FakePPO.fail_learn = True
    else:
        FakePPO.fail_save = True

    with pytest.raises(RuntimeError, match=message):
        run_with_fakes(
            make_config(num_envs=2),
            tmp_path / failure,
            env_factory=environments,
            subproc_factory=VecFactory(),
        )

    assert environments.created
    assert all(env.closed for env in environments.created)
    assert all(env.close_calls == 1 for env in environments.created)


class FailingEvalVecFactory:
    def __init__(self) -> None:
        self.partial: FakeEnv | None = None

    def __call__(self, env_fns: list[Callable[[], FakeEnv]]) -> FakeVecEnv:
        self.partial = env_fns[0]()
        raise RuntimeError("eval construction failed")


def test_eval_construction_failure_closes_train_and_partial_eval_env(tmp_path: Path) -> None:
    environments = EnvFactory()
    dummy = FailingEvalVecFactory()

    with pytest.raises(RuntimeError, match="eval construction failed"):
        run_training(
            make_config(num_envs=2),
            smoke=True,
            run_dir=tmp_path / "eval-failure",
            env_factory=environments,
            ppo_factory=FakePPO,
            dummy_vec_env_factory=dummy,
            subproc_vec_env_factory=VecFactory(),
            checkpoint_callback_factory=CallbackFactory(),
            eval_callback_factory=CallbackFactory(),
            reward_callback_factory=RewardCallbackFactory(),
        )

    assert dummy.partial is not None
    assert dummy.partial.unwrapped.closed is True
    assert all(env.closed for env in environments.created)
    assert all(env.close_calls == 1 for env in environments.created)
