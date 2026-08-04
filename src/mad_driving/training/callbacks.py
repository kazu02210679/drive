"""Stable-Baselines3 callbacks for training diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Final

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

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


class SeededEvalCallback(EvalCallback):
    """Restart one validation environment's episode-seed sequence before each evaluation."""

    validation_episode_seed: int

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
        super().__init__(
            eval_env,
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
        if self.eval_env.num_envs != 1:
            raise ValueError("SeededEvalCallback requires exactly one validation environment")
        self.validation_episode_seed = int(validation_episode_seed)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            assigned_seeds = tuple(self.eval_env.seed(self.validation_episode_seed))
            if assigned_seeds != (self.validation_episode_seed,):
                raise RuntimeError("Validation environment did not accept its fixed episode seed")
        return super()._on_step()


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
