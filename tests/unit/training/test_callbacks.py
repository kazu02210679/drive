import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from mad_driving.config.models import CurriculumConfig
from mad_driving.training import callbacks as callbacks_module
from mad_driving.training.callbacks import CurriculumEvalCallback, RewardComponentsCallback
from mad_driving.training.curriculum import (
    CurriculumController,
    CurriculumState,
    read_curriculum_state,
)

REWARD_COMPONENT_KEYS = (
    "progress_reward",
    "arrival_reward",
    "collision_penalty",
    "near_miss_penalty",
    "offroad_penalty",
    "rule_violation_penalty",
    "jerk_penalty",
    "unnecessary_brake_penalty",
    "standstill_penalty",
    "shield_intervention_penalty",
)


class RecordingLogger:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.mean_calls: list[tuple[str, float]] = []

    def record(self, key: str, value: float) -> None:
        self.values[key] = value

    def record_mean(self, key: str, value: float) -> None:
        self.mean_calls.append((key, value))
        self.values[key] = value


class FakeModel:
    def __init__(self) -> None:
        self.logger = RecordingLogger()
        self.num_timesteps = 0


class SeedRecordingEnv(gym.Env[NDArray[np.float32], int]):
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self) -> None:
        self.reset_seeds: list[int | None] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        super().reset(seed=seed)
        del options
        self.reset_seeds.append(seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        assert self.action_space.contains(action)
        return np.zeros(4, dtype=np.float32), 0.0, True, False, {}


class CurriculumOutcomeEnv(gym.Env[NDArray[np.float32], int]):
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self, *, scenario_success: bool, collision_occurred: bool) -> None:
        self.scenario_success = scenario_success
        self.collision_occurred = collision_occurred
        self.reset_seeds: list[int | None] = []
        self.levels: list[int] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        super().reset(seed=seed)
        del options
        self.reset_seeds.append(seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        assert self.action_space.contains(action)
        return np.zeros(4, dtype=np.float32), 0.0, True, False, {
            "scenario_success": self.scenario_success,
            "collision_occurred": self.collision_occurred,
        }

    def set_difficulty_level(self, level: int) -> None:
        self.levels.append(level)


def initialized_callback() -> tuple[RewardComponentsCallback, FakeModel]:
    callback = RewardComponentsCallback()
    model = FakeModel()
    callback.init_callback(model)  # type: ignore[arg-type]
    return callback, model


def test_records_every_reward_component_from_vector_infos() -> None:
    callback, model = initialized_callback()
    first = {key: float(index) for index, key in enumerate(REWARD_COMPONENT_KEYS, start=1)}
    second = {key: value + 2.0 for key, value in first.items()}
    callback.locals = {
        "infos": [
            {"reward_components": first},
            {"reward_components": second},
        ]
    }

    assert callback._on_step() is True

    assert model.logger.values == {
        f"reward/{key}": (first[key] + second[key]) / 2.0 for key in REWARD_COMPONENT_KEYS
    }
    assert model.logger.mean_calls == [
        (f"reward/{key}", (first[key] + second[key]) / 2.0) for key in REWARD_COMPONENT_KEYS
    ]


def test_ignores_missing_and_malformed_reward_info_without_stopping_learning() -> None:
    callback, model = initialized_callback()
    callback.locals = {
        "infos": [
            {},
            None,
            {"reward_components": "not-a-mapping"},
            {"reward_components": {"progress_reward": math.inf}},
            {"reward_components": {"arrival_reward": True}},
            {"reward_components": {"unknown": 3.0}},
            {"reward_components": {"progress_reward": 1.25}},
        ]
    }

    assert callback._on_step() is True
    assert model.logger.values == {"reward/progress_reward": 1.25}


def test_ignores_non_sequence_infos() -> None:
    callback, model = initialized_callback()
    callback.locals = {"infos": {"reward_components": {"progress_reward": 1.0}}}

    assert callback._on_step() is True
    assert model.logger.values == {}


def test_real_sb3_callback_list_attaches_model_logger_and_saves(
    tmp_path: Path,
) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv

    train_env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    eval_env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            device="cpu",
        )
        model.set_logger(configure(str(tmp_path / "logger"), []))
        reward_callback = RewardComponentsCallback()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(tmp_path / "best"),
            eval_freq=1,
            n_eval_episodes=1,
        )
        callbacks = CallbackList([reward_callback, eval_callback])

        callbacks.init_callback(model)
        model_path = tmp_path / "callback_model.zip"
        model.save(model_path)

        assert reward_callback.model is model
        assert reward_callback.logger is model.logger
        assert eval_callback.model is model
        assert eval_callback.logger is model.logger
        assert model_path.is_file()
    finally:
        eval_env.close()
        train_env.close()


def test_seeded_eval_callback_restarts_the_same_seed_sequence_before_every_eval(
    tmp_path: Path,
) -> None:
    train_env = DummyVecEnv([lambda: SeedRecordingEnv()])
    evaluation_environment = SeedRecordingEnv()
    eval_env = DummyVecEnv([lambda: evaluation_environment])
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            device="cpu",
        )
        model.set_logger(configure(str(tmp_path / "seeded-eval-logger"), []))
        callback_class = callbacks_module.SeededEvalCallback
        callback = callback_class(
            eval_env,
            validation_episode_seed=73,
            eval_freq=2,
            n_eval_episodes=1,
            deterministic=True,
            render=False,
            verbose=0,
        )

        model.learn(total_timesteps=4, callback=callback)

        assert evaluation_environment.reset_seeds == [73, None, 73, None]
    finally:
        eval_env.close()
        train_env.close()


def test_seeded_eval_callback_rejects_invalid_validation_seed() -> None:
    eval_env = DummyVecEnv([lambda: SeedRecordingEnv()])
    try:
        callback_class = callbacks_module.SeededEvalCallback
        for invalid_seed in (-1, True):
            try:
                callback_class(eval_env, validation_episode_seed=invalid_seed)
            except ValueError as error:
                assert "validation_episode_seed" in str(error)
            else:
                raise AssertionError(f"invalid validation seed was accepted: {invalid_seed!r}")
    finally:
        eval_env.close()


def run_curriculum_evaluation(
    tmp_path: Path,
    *,
    scenario_success: bool,
    collision_occurred: bool,
    config: CurriculumConfig,
) -> tuple[
    CurriculumEvalCallback,
    CurriculumOutcomeEnv,
    CurriculumOutcomeEnv,
    Path,
]:
    training_environment = CurriculumOutcomeEnv(
        scenario_success=False,
        collision_occurred=False,
    )
    evaluation_environment = CurriculumOutcomeEnv(
        scenario_success=scenario_success,
        collision_occurred=collision_occurred,
    )
    train_env = DummyVecEnv([lambda: training_environment])
    eval_env = DummyVecEnv([lambda: evaluation_environment])
    state_path = tmp_path / "curriculum_state.yaml"
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        device="cpu",
    )
    model.set_logger(configure(str(tmp_path / "curriculum-eval-logger"), []))
    initial_level = config.fixed_level if config.mode == "fixed" else config.initial_level
    callback = CurriculumEvalCallback(
        eval_env,
        validation_episode_seed=73,
        controller=CurriculumController(
            config,
            CurriculumState(initial_level, 0, 0),
        ),
        curriculum_state_path=state_path,
        eval_freq=4,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=0,
    )
    try:
        model.learn(total_timesteps=4, callback=callback)
    except BaseException:
        eval_env.close()
        train_env.close()
        raise
    eval_env.close()
    train_env.close()
    return callback, training_environment, evaluation_environment, state_path


def test_curriculum_callback_persists_and_broadcasts_changed_level_after_eval(
    tmp_path: Path,
) -> None:
    callback, training_environment, evaluation_environment, state_path = (
        run_curriculum_evaluation(
            tmp_path,
            scenario_success=True,
            collision_occurred=False,
            config=CurriculumConfig(
                mode="automatic",
                consecutive_evaluations=1,
            ),
        )
    )

    expected = CurriculumState(level=1, consecutive_passes=0, evaluations=1)
    assert callback.controller.state == expected
    assert read_curriculum_state(state_path) == expected
    assert training_environment.levels == [1]
    assert evaluation_environment.levels == [1]
    assert evaluation_environment.reset_seeds == [73, None]


def test_curriculum_callback_persists_without_broadcast_when_level_is_unchanged(
    tmp_path: Path,
) -> None:
    callback, training_environment, evaluation_environment, state_path = (
        run_curriculum_evaluation(
            tmp_path,
            scenario_success=False,
            collision_occurred=False,
            config=CurriculumConfig(mode="automatic"),
        )
    )

    expected = CurriculumState(level=0, consecutive_passes=0, evaluations=1)
    assert callback.controller.state == expected
    assert read_curriculum_state(state_path) == expected
    assert training_environment.levels == []
    assert evaluation_environment.levels == []


@pytest.mark.parametrize(
    "info",
    [
        {"collision_occurred": False},
        {"scenario_success": True},
        {"scenario_success": 1, "collision_occurred": False},
        {"scenario_success": True, "collision_occurred": 0},
    ],
)
def test_curriculum_callback_rejects_missing_or_non_boolean_terminal_metrics(
    tmp_path: Path,
    info: dict[str, object],
) -> None:
    eval_env = DummyVecEnv([lambda: SeedRecordingEnv()])
    callback = CurriculumEvalCallback(
        eval_env,
        validation_episode_seed=73,
        controller=CurriculumController(
            CurriculumConfig(mode="automatic"),
            CurriculumState(0, 0, 0),
        ),
        curriculum_state_path=tmp_path / "curriculum_state.yaml",
    )
    try:
        with pytest.raises(ValueError, match="scenario_success|collision_occurred"):
            callback._log_success_callback(
                {"info": info, "done": True},
                {},
            )
    finally:
        eval_env.close()


def test_curriculum_callback_ignores_nonterminal_info() -> None:
    eval_env = DummyVecEnv([lambda: SeedRecordingEnv()])
    callback = CurriculumEvalCallback(
        eval_env,
        validation_episode_seed=73,
        controller=CurriculumController(
            CurriculumConfig(mode="automatic"),
            CurriculumState(0, 0, 0),
        ),
        curriculum_state_path=Path("unused-until-scheduled.yaml"),
    )
    try:
        callback._log_success_callback({"info": {}, "done": False}, {})
        assert callback.terminal_records == ()
    finally:
        eval_env.close()
