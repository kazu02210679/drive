"""Stable-Baselines3 callbacks for training diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

from mad_driving.training.curriculum import (
    CurriculumController,
    CurriculumState,
    write_checkpoint_curriculum_state,
    write_curriculum_state,
)

REWARD_COMPONENT_KEYS: Final = (
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


class _AbsoluteTimestepSchedule:
    """One repeating deadline sequence anchored at absolute model timestep zero."""

    interval_timesteps: int
    _next_deadline: int | None

    def __init__(self, interval_timesteps: int, *, name: str) -> None:
        if (
            isinstance(interval_timesteps, bool)
            or not isinstance(interval_timesteps, Integral)
            or interval_timesteps <= 0
        ):
            raise ValueError(f"{name} must be a positive non-bool integer")
        self.interval_timesteps = int(interval_timesteps)
        self._next_deadline: int | None = None

    def initialize(self, current_timestep: int) -> None:
        current = self._validated_timestep(current_timestep)
        self._next_deadline = (
            current // self.interval_timesteps + 1
        ) * self.interval_timesteps

    def consume_if_due(self, current_timestep: int) -> bool:
        current = self._validated_timestep(current_timestep)
        if self._next_deadline is None:
            self.initialize(current)
            return False
        if current < self._next_deadline:
            return False
        while self._next_deadline <= current:
            self._next_deadline += self.interval_timesteps
        return True

    @staticmethod
    def _validated_timestep(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError("model num_timesteps must be a non-negative non-bool integer")
        return int(value)


class SeededEvalCallback(EvalCallback):
    """Restart one validation environment's episode-seed sequence before each evaluation."""

    validation_episode_seed: int
    _evaluation_schedule: _AbsoluteTimestepSchedule

    def __init__(
        self,
        eval_env: gym.Env[Any, Any] | VecEnv,
        *,
        validation_episode_seed: int,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ) -> None:
        if (
            isinstance(validation_episode_seed, bool)
            or not isinstance(validation_episode_seed, Integral)
            or validation_episode_seed < 0
        ):
            raise ValueError("validation_episode_seed must be a non-negative non-bool integer")
        self._evaluation_schedule = _AbsoluteTimestepSchedule(
            eval_freq,
            name="eval_freq",
        )
        super().__init__(
            eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=1,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        if self.eval_env.num_envs != 1:
            raise ValueError("SeededEvalCallback requires exactly one validation environment")
        self.validation_episode_seed = int(validation_episode_seed)

    def _on_training_start(self) -> None:
        super()._on_training_start()
        self._evaluation_schedule.initialize(self.num_timesteps)

    def _evaluation_due(self) -> bool:
        return self._evaluation_schedule.consume_if_due(self.num_timesteps)

    def _run_scheduled_evaluation(self) -> bool:
        assigned_seeds = tuple(self.eval_env.seed(self.validation_episode_seed))
        if assigned_seeds != (self.validation_episode_seed,):
            raise RuntimeError("Validation environment did not accept its fixed episode seed")
        return super()._on_step()

    def _on_step(self) -> bool:
        if not self._evaluation_due():
            return True
        return self._run_scheduled_evaluation()


class CurriculumCheckpointCallback(CheckpointCallback):
    """Bind the current curriculum state to every periodic model checkpoint."""

    _checkpoint_schedule: _AbsoluteTimestepSchedule

    def __init__(
        self,
        *,
        controller: CurriculumController,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ) -> None:
        if not isinstance(controller, CurriculumController):
            raise TypeError("controller must be a CurriculumController")
        super().__init__(
            save_freq=1,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose,
        )
        self.controller = controller
        self._checkpoint_schedule = _AbsoluteTimestepSchedule(
            save_freq,
            name="save_freq",
        )

    def _on_training_start(self) -> None:
        super()._on_training_start()
        self._checkpoint_schedule.initialize(self.num_timesteps)

    def _on_step(self) -> bool:
        scheduled = self._checkpoint_schedule.consume_if_due(self.num_timesteps)
        if not scheduled:
            return True
        continue_training = super()._on_step()
        write_checkpoint_curriculum_state(
            self.controller.state,
            self._checkpoint_path(extension="zip"),
        )
        return continue_training


class CurriculumEvalCallback(SeededEvalCallback):
    """Observe typed validation outcomes and broadcast reset-boundary level changes."""

    def __init__(
        self,
        eval_env: gym.Env[Any, Any] | VecEnv,
        *,
        validation_episode_seed: int,
        controller: CurriculumController,
        curriculum_state_path: Path,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ) -> None:
        if not isinstance(controller, CurriculumController):
            raise TypeError("controller must be a CurriculumController")
        super().__init__(
            eval_env,
            validation_episode_seed=validation_episode_seed,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        self.controller = controller
        self.curriculum_state_path = Path(curriculum_state_path)
        self._terminal_records: list[tuple[bool, bool]] = []

    @property
    def terminal_records(self) -> tuple[tuple[bool, bool], ...]:
        """Return the records captured during the current scheduled evaluation."""

        return tuple(self._terminal_records)

    def _log_success_callback(
        self,
        locals_: dict[str, Any],
        globals_: dict[str, Any],
    ) -> None:
        super()._log_success_callback(locals_, globals_)
        done = locals_.get("done")
        if not isinstance(done, bool | np.bool_):
            raise ValueError("validation done must be a boolean")
        if not bool(done):
            return
        info = locals_.get("info")
        if not isinstance(info, Mapping):
            raise ValueError("terminal validation info must be a mapping")
        scenario_success = info.get("scenario_success")
        collision_occurred = info.get("collision_occurred")
        if type(scenario_success) is not bool:
            raise ValueError("terminal scenario_success must be a boolean")
        if type(collision_occurred) is not bool:
            raise ValueError("terminal collision_occurred must be a boolean")
        self._terminal_records.append((scenario_success, collision_occurred))

    def _on_step(self) -> bool:
        scheduled = self._evaluation_due()
        if not scheduled:
            return True
        previous_state: CurriculumState | None = None
        previous_best_mean_reward = self.best_mean_reward
        self._terminal_records = []
        previous_state = self.controller.state
        continue_training = self._run_scheduled_evaluation()
        if len(self._terminal_records) != self.n_eval_episodes:
            raise RuntimeError(
                "scheduled validation did not produce exactly one terminal curriculum record "
                "per episode"
            )
        state = self.controller.observe(
            "validation",
            successes=sum(success for success, _collision in self._terminal_records),
            collisions=sum(collision for _success, collision in self._terminal_records),
            episodes=len(self._terminal_records),
        )
        write_curriculum_state(state, self.curriculum_state_path)
        if (
            self.best_mean_reward > previous_best_mean_reward
            and self.best_model_save_path is not None
        ):
            write_checkpoint_curriculum_state(
                state,
                Path(self.best_model_save_path) / "best_model.zip",
            )
        if previous_state is not None and state.level != previous_state.level:
            self.training_env.env_method("set_difficulty_level", state.level)
            self.eval_env.env_method("set_difficulty_level", state.level)
        return continue_training


class RewardComponentsCallback(BaseCallback):
    """Record finite reward components averaged across envs and logger intervals."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not isinstance(infos, Sequence) or isinstance(infos, str | bytes):
            return True

        values: dict[str, list[float]] = {key: [] for key in REWARD_COMPONENT_KEYS}
        for info in infos:
            if not isinstance(info, Mapping):
                continue
            components = info.get("reward_components")
            if not isinstance(components, Mapping):
                continue
            for key in REWARD_COMPONENT_KEYS:
                value = components.get(key)
                if isinstance(value, bool) or not isinstance(value, Real):
                    continue
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    values[key].append(numeric_value)

        for key, samples in values.items():
            if samples:
                self.logger.record_mean(f"reward/{key}", sum(samples) / len(samples))
        return True
