import math
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import cloudpickle
import pytest
import yaml

from mad_driving.config.models import AppConfig
from mad_driving.training import train as train_module
from mad_driving.training.train import TrainingResult, run_training


class FakeEnv:
    def __init__(self, identifier: int) -> None:
        self.identifier = identifier
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1


class EnvFactory:
    def __init__(self) -> None:
        self.created: list[FakeEnv] = []

    def __call__(self, config: AppConfig) -> FakeEnv:
        del config
        env = FakeEnv(len(self.created))
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
    resume_num_timesteps: ClassVar[int] = 12_500

    def __init__(self, policy: str, env: FakeVecEnv, **kwargs: Any) -> None:
        if type(self).fail_init:
            raise RuntimeError("PPO construction failed")
        self.init_kwargs = {"policy": policy, "env": env, **kwargs}
        self.learn_kwargs: dict[str, Any] | None = None
        self.saved_path: Path | None = None
        self.num_timesteps = 0
        type(self).instances.append(self)

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
    def write_checkpoint(path: Path, marker: str) -> None:
        with zipfile.ZipFile(path, "w") as checkpoint:
            checkpoint.writestr("marker.txt", marker)


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
        "tensorboard_log": str(run_dir / "tensorboard"),
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


def test_failed_best_validation_preserves_existing_best_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "stale-best"
    best_checkpoint = run_dir / "checkpoints" / "best_model.zip"
    best_checkpoint.parent.mkdir(parents=True)
    best_checkpoint.write_bytes(b"stale")
    environments = EnvFactory()

    with pytest.raises(FileNotFoundError, match="Best checkpoint"):
        run_with_fakes(
            make_config(),
            run_dir,
            env_factory=environments,
            eval_factory=CallbackFactory(),
        )

    assert best_checkpoint.read_bytes() == b"stale"
    assert all(env.close_calls == 1 for env in environments.created)


def test_scheduled_evaluation_overwrites_stale_best_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "fresh-best"
    best_checkpoint = run_dir / "checkpoints" / "best_model.zip"
    best_checkpoint.parent.mkdir(parents=True)
    best_checkpoint.write_bytes(b"stale")

    result, *_ = run_with_fakes(make_config(), run_dir)

    assert result.best_checkpoint == best_checkpoint
    assert zipfile.is_zipfile(best_checkpoint)
    with zipfile.ZipFile(best_checkpoint) as checkpoint:
        assert checkpoint.read("marker.txt") == b"best"


def test_failed_final_validation_preserves_existing_final_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "stale-final"
    final_checkpoint = run_dir / "checkpoints" / "final_model.zip"
    final_checkpoint.parent.mkdir(parents=True)
    final_checkpoint.write_bytes(b"stale")
    FakePPO.skip_save = True
    environments = EnvFactory()

    with pytest.raises(FileNotFoundError, match="Final checkpoint"):
        run_with_fakes(make_config(), run_dir, env_factory=environments)

    assert final_checkpoint.read_bytes() == b"stale"
    assert all(env.close_calls == 1 for env in environments.created)


def seed_existing_checkpoints(run_dir: Path) -> dict[Path, bytes]:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "final": checkpoints_dir / "final_model.zip",
        "best": checkpoints_dir / "best_model.zip",
        "periodic": checkpoints_dir / "ppo_checkpoint_17500_steps.zip",
    }
    for marker, path in paths.items():
        FakePPO.write_checkpoint(path, f"old-{marker}")
    return {path: path.read_bytes() for path in paths.values()}


def assert_no_invocation_staging(run_dir: Path) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    assert [path for path in checkpoints_dir.iterdir() if path.name.startswith(".training-")] == []


class FailingEnvFactory:
    def __call__(self, config: AppConfig) -> FakeEnv:
        del config
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
def test_failed_run_preserves_all_existing_checkpoint_artifacts(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    run_dir = tmp_path / failure
    original_artifacts = seed_existing_checkpoints(run_dir)
    resume_from = run_dir / "checkpoints" / "final_model.zip"
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
            resume_from=resume_from,
            env_factory=env_factory,
            dummy_factory=VecFactory(),
            eval_factory=eval_factory,
        )

    assert {path: path.read_bytes() for path in original_artifacts} == original_artifacts
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
    assert staging_dir.parent == run_dir / "checkpoints"
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
def test_promotion_failure_restores_all_originals_and_removes_new_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: int,
) -> None:
    run_dir = tmp_path / f"replace-{failure_at}"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    original_paths = [
        checkpoints_dir / "ppo_checkpoint_2500_steps.zip",
        checkpoints_dir / "best_model.zip",
        checkpoints_dir / "final_model.zip",
    ]
    for index, path in enumerate(original_paths):
        path.write_bytes(f"original-{index}".encode())
    originals = {path: path.read_bytes() for path in original_paths}
    new_periodic = checkpoints_dir / "ppo_checkpoint_5000_steps.zip"
    real_replace = train_module.os.replace
    replace_calls = 0

    def fail_selected_replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == failure_at:
            raise OSError(f"replacement {failure_at} failed")
        real_replace(source, destination)

    monkeypatch.setattr(train_module.os, "replace", fail_selected_replace)

    with pytest.raises(OSError, match=f"replacement {failure_at} failed"):
        run_with_fakes(
            make_config(checkpoint_interval_steps=2_500),
            run_dir,
        )

    assert {path: path.read_bytes() for path in original_paths} == originals
    assert not new_periodic.exists()
    assert_no_invocation_staging(run_dir)


@pytest.mark.parametrize("resume_name", ["final_model.zip", "best_model.zip"])
def test_same_run_checkpoint_remains_readable_through_ppo_load(
    tmp_path: Path,
    resume_name: str,
) -> None:
    run_dir = tmp_path / resume_name.removesuffix(".zip")
    resume_path = run_dir / "checkpoints" / resume_name
    resume_path.parent.mkdir(parents=True)
    FakePPO.write_checkpoint(resume_path, "resume-source")
    source_bytes = resume_path.read_bytes()

    result, *_ = run_with_fakes(
        make_config(),
        run_dir,
        resume_from=resume_path,
    )

    assert FakePPO.load_calls[0]["path"] == resume_path
    assert FakePPO.load_calls[0]["source_bytes"] == source_bytes
    assert result.final_checkpoint.is_file()
    assert result.best_checkpoint.is_file()
    assert_no_invocation_staging(run_dir)


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
    vector_env = RuntimeProcessVecEnv([lambda: FakeEnv(0)])
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


def test_resume_loads_ppo_with_train_env_and_keeps_timesteps(tmp_path: Path) -> None:
    resume_from = tmp_path / "source.zip"
    resume_from.write_bytes(b"checkpoint")

    FakePPO.rollout_overshoot = 96
    result, *_ = run_with_fakes(
        make_config(),
        tmp_path / "resume",
        resume_from=resume_from,
    )

    assert len(FakePPO.load_calls) == 1
    assert len(FakePPO.instances) == 1
    model = FakePPO.instances[0]
    assert FakePPO.load_calls[0] == {
        "path": resume_from,
        "source_bytes": b"checkpoint",
        "env": model.init_kwargs["env"],
        "device": "cpu",
        "custom_objects": None,
        "print_system_info": False,
        "force_reset": True,
        "tensorboard_log": str(tmp_path / "resume" / "tensorboard"),
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
