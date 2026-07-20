from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from mad_driving.config.models import AppConfig
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

    def __init__(self, policy: str, env: FakeVecEnv, **kwargs: Any) -> None:
        self.init_kwargs = {"policy": policy, "env": env, **kwargs}
        self.learn_kwargs: dict[str, Any] | None = None
        self.saved_path: Path | None = None
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.load_calls = []
        cls.fail_learn = False
        cls.fail_save = False

    @classmethod
    def load(cls, path: Path, *, env: FakeVecEnv, device: str) -> "FakePPO":
        cls.load_calls.append({"path": path, "env": env, "device": device})
        model = cls.__new__(cls)
        model.init_kwargs = {"env": env}
        model.learn_kwargs = None
        model.saved_path = None
        cls.instances.append(model)
        return model

    def learn(self, **kwargs: Any) -> "FakePPO":
        self.learn_kwargs = kwargs
        if type(self).fail_learn:
            raise RuntimeError("learn failed")
        return self

    def save(self, path: Path) -> None:
        self.saved_path = Path(path)
        if type(self).fail_save:
            raise RuntimeError("save failed")
        self.saved_path.write_bytes(b"fake checkpoint")


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
    checkpoint_factory: CallbackFactory | None = None,
    eval_factory: CallbackFactory | None = None,
    reward_factory: RewardCallbackFactory | None = None,
) -> tuple[TrainingResult, EnvFactory, VecFactory, CallbackFactory, CallbackFactory]:
    environments = env_factory or EnvFactory()
    dummy = dummy_factory or VecFactory()
    checkpoint = checkpoint_factory or CallbackFactory()
    evaluation = eval_factory or CallbackFactory()
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

    result, environments, dummy, _, _ = run_with_fakes(config, run_dir)

    model = FakePPO.instances[0]
    train_env = dummy.created[0]
    eval_env = dummy.created[1]
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
    assert environments.created[0] is not environments.created[1]
    assert all(env.closed for env in environments.created)
    assert all(vec.closed for vec in dummy.created)
    assert result == TrainingResult(
        run_dir=run_dir,
        final_checkpoint=run_dir / "checkpoints" / "final_model.zip",
        best_checkpoint=run_dir / "checkpoints" / "best_model.zip",
        timesteps=5_000,
    )
    assert result.final_checkpoint.is_file()


def test_normal_training_uses_500_000_timesteps(tmp_path: Path) -> None:
    result, *_ = run_with_fakes(make_config(), tmp_path / "normal", smoke=False)

    assert FakePPO.instances[0].learn_kwargs is not None
    assert FakePPO.instances[0].learn_kwargs["total_timesteps"] == 500_000
    assert result.timesteps == 500_000


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
    evaluation = CallbackFactory()
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
    assert evaluation.calls[0]["eval_freq"] == 2_000


def test_callback_frequencies_have_a_minimum_of_one(tmp_path: Path) -> None:
    checkpoint = CallbackFactory()
    evaluation = CallbackFactory()
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


def test_subprocess_failure_closes_partial_resources_and_rebuilds_equal_dummy_count(
    tmp_path: Path,
) -> None:
    dummy = VecFactory()
    environments = EnvFactory()

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
    assert [vec.num_envs for vec in dummy.created] == [3, 1]
    assert len(environments.created) == 5
    assert all(env.closed for env in environments.created)


def test_resume_loads_ppo_with_train_env_and_keeps_timesteps(tmp_path: Path) -> None:
    resume_from = tmp_path / "source.zip"
    resume_from.write_bytes(b"checkpoint")

    run_with_fakes(
        make_config(),
        tmp_path / "resume",
        resume_from=resume_from,
    )

    assert len(FakePPO.load_calls) == 1
    assert len(FakePPO.instances) == 1
    model = FakePPO.instances[0]
    assert FakePPO.load_calls[0] == {
        "path": resume_from,
        "env": model.init_kwargs["env"],
        "device": "cpu",
    }
    assert model.learn_kwargs is not None
    assert model.learn_kwargs["reset_num_timesteps"] is False


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
