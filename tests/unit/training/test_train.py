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
import numpy as np
import pytest
import yaml
from gymnasium import spaces
from stable_baselines3.common.utils import ConstantSchedule, FloatSchedule

from mad_driving.config.models import AppConfig
from mad_driving.scenarios import EnvironmentRole
from mad_driving.training import RunMetadata, sha256_file
from mad_driving.training import metadata as metadata_module
from mad_driving.training import ownership as ownership_module
from mad_driving.training import train as train_module
from mad_driving.training.train import TrainingResult, run_training


class FakeEnv:
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

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1


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
        self.save_freq = kwargs["save_freq"]
        self.save_path = Path(kwargs["save_path"])
        self.name_prefix = kwargs["name_prefix"]

    def on_fake_training(self, model: "FakePPO", produced_timesteps: int) -> None:
        train_env = model.init_kwargs["env"]
        callback_calls = math.ceil(produced_timesteps / train_env.num_envs)
        start_timesteps = model.num_timesteps - produced_timesteps
        for call in range(self.save_freq, callback_calls + 1, self.save_freq):
            checkpoint_timesteps = start_timesteps + call * train_env.num_envs
            model.write_checkpoint(
                self.save_path / f"{self.name_prefix}_{checkpoint_timesteps}_steps.zip",
                "periodic",
            )


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
        self.eval_env = kwargs["eval_env"]
        self.eval_freq = kwargs["eval_freq"]
        self.best_model_save_path = Path(kwargs["best_model_save_path"])

    def on_fake_training(self, model: "FakePPO", produced_timesteps: int) -> None:
        train_env = model.init_kwargs["env"]
        callback_calls = math.ceil(produced_timesteps / train_env.num_envs)
        if callback_calls >= self.eval_freq:
            model.write_checkpoint(self.best_model_save_path / "best_model.zip", "best")


class EvalCallbackFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.instances: list[FakeEvalCallback] = []

    def __call__(self, **kwargs: Any) -> FakeEvalCallback:
        self.calls.append(kwargs)
        callback = FakeEvalCallback(**kwargs)
        self.instances.append(callback)
        return callback


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
        path: Path,
        env: FakeVecEnv | None = None,
        device: str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> "FakePPO":
        source_bytes = Path(path).read_bytes()
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
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as checkpoint:
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


@dataclass(frozen=True)
class CompatibleSourceRun:
    run_dir: Path
    checkpoint: Path


def seed_compatible_source_run(
    tmp_path: Path,
    **ppo_overrides: Any,
) -> CompatibleSourceRun:
    index = 0
    while (tmp_path / f"source-run-{index}").exists():
        index += 1
    run_dir = tmp_path / f"source-run-{index}"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    config = make_config()
    resolved_config = config.model_dump(mode="json")
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    metadata = RunMetadata(resolved_config=resolved_config)
    metadata_module.write_run_metadata(metadata, run_dir / "run_metadata.json")
    checkpoint = checkpoints_dir / "final_model.zip"
    FakePPO.write_checkpoint(
        checkpoint,
        "resume-source",
        {**FakePPO.default_contract(), **ppo_overrides},
    )
    return CompatibleSourceRun(run_dir=run_dir, checkpoint=checkpoint)


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
        ppo_factory=FakePPO,
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


def test_run_training_uses_only_configured_ppo_values_and_closes_envs(tmp_path: Path) -> None:
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
    assert environments.created[0] is not environments.created[1]
    assert all(env.closed for env in environments.created)
    assert all(env.close_calls == 1 for env in environments.created)
    assert all(vec.closed for vec in [*dummy.created, eval_env])
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
    assert metadata == {
        "research_contract_version": 2,
        "observation_schema_version": 1,
        "observation_shape": [24],
        "observation_dtype": "float32",
        "action_schema_version": 1,
        "action_count": 4,
        "action_order": ["KEEP", "SLOW", "PREPARE_STOP", "STOP"],
        "resolved_config": config.model_dump(mode="json"),
        "resume": None,
    }


def test_callback_frequencies_are_scaled_by_num_envs(tmp_path: Path) -> None:
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
        subproc_factory=VecFactory(),
        checkpoint_factory=checkpoint,
        eval_factory=evaluation,
    )

    assert checkpoint.calls[0]["save_freq"] == 2_500
    assert evaluation.calls[0]["eval_freq"] == 1_250


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
            eval_factory=CallbackFactory(),
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
    eval_factory: EvalCallbackFactory | CallbackFactory = EvalCallbackFactory()
    if failure == "environment":
        env_factory = FailingEnvFactory()
    elif failure == "ppo":
        FakePPO.fail_load = True
    elif failure == "learn":
        FakePPO.fail_learn = True
    elif failure == "save":
        FakePPO.fail_save = True
    elif failure == "best_validation":
        eval_factory = CallbackFactory()
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
    assert FailingSubprocVecEnv.latest.envs[0].closed is True
    assert FailingSubprocVecEnv.latest.envs[0].close_calls == 1
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

    def kill(self) -> None:
        self.kill_calls += 1
        if self.exits_on_kill:
            self.alive = False

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
) -> None:
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
    assert terminated.join_calls == [1.0]
    assert killed.terminate_calls == 1
    assert killed.kill_calls == 1
    assert killed.join_calls == [1.0, 1.0]
    assert already_dead.terminate_calls == 0
    assert already_dead.kill_calls == 0
    assert already_dead.join_calls == []
    assert all(process.is_alive() is False for process in partial.processes)
    assert all(process.is_alive_calls >= 2 for process in partial.processes)
    assert dummy.created == []


def test_subprocess_cleanup_refuses_fallback_when_worker_survives(tmp_path: Path) -> None:
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
    assert process.join_calls == [1.0, 1.0]
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
    def __init__(self, process: FakeProcess) -> None:
        super().__init__()
        self.process = process
        self.sent: list[tuple[str, None]] = []

    def send(self, message: tuple[str, None]) -> None:
        self.sent.append(message)
        if message == ("close", None):
            self.process.alive = False


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
        self.remotes = (GracefulRemote(self.processes[0]),)
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


def test_successful_training_confirms_subprocess_evaluation_worker_exited(
    tmp_path: Path,
) -> None:
    subproc = RuntimeProcessVecFactory()

    run_with_fakes(
        make_config(),
        tmp_path / "bounded-close",
        subproc_factory=subproc,
    )

    evaluation = subproc.created[0]
    process = evaluation.processes[0]
    remote = evaluation.remotes[0]
    assert remote.sent == [("close", None)]
    assert remote.close_calls == 1
    assert process.join_calls == [1.0]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.is_alive() is False
    assert evaluation.closed is True


def test_runtime_cleanup_continues_to_kill_after_join_and_terminate_fail() -> None:
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

    train_module._close_vector_env(vector_env)

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.is_alive() is False


def test_worker_cleanup_failure_replaces_success_with_explicit_error(tmp_path: Path) -> None:
    subproc = RuntimeProcessVecFactory(
        graceful=False,
        exits_on_terminate=False,
        exits_on_kill=False,
    )

    with pytest.raises(RuntimeError, match="worker cleanup could not be confirmed"):
        run_with_fakes(
            make_config(),
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
            make_config(),
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
    assert [(env.role, env.worker_index) for env in train_envs] == [
        ("train", 0),
        ("train", 1),
        ("train", 2),
    ]
    assert [(env.role, env.worker_index) for env in eval_envs] == [("validation", 0)]
    configs = [env.config for env in (*train_envs, *eval_envs)]
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
    assert FakePPO.load_calls[0]["path"] == source.checkpoint.resolve()
    assert FakePPO.instances[0].learn_kwargs is not None
    assert FakePPO.instances[0].learn_kwargs["reset_num_timesteps"] is False
    assert FakePPO.instances[0].num_timesteps == 17_596
    assert result.timesteps == 5_096
    assert source_tree_bytes(source.run_dir) == source_before


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


def test_resume_detects_checkpoint_change_during_load_before_destination_or_learning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = seed_compatible_source_run(tmp_path)
    destination = tmp_path / "changed-source"
    original_load = FakePPO.load.__func__

    def changing_load(cls: type[FakePPO], path: Path, **kwargs: Any) -> FakePPO:
        model = original_load(cls, path, **kwargs)
        path.write_bytes(b"changed-during-load")
        return model

    monkeypatch.setattr(FakePPO, "load", classmethod(changing_load))

    with pytest.raises(ValueError, match="changed"):
        run_with_fakes(make_config(), destination, resume_from=source.checkpoint)

    assert not destination.exists()
    assert FakePPO.instances[0].learn_kwargs is None


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
    assert FakePPO.load_calls[0] == {
        "path": source.checkpoint.resolve(),
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
    assert dummy.partial.closed is True
    assert all(env.closed for env in environments.created)
    assert all(env.close_calls == 1 for env in environments.created)
