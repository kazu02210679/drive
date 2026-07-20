"""Stable-Baselines3 callbacks for training diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Final

from stable_baselines3.common.callbacks import BaseCallback

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


class RewardComponentsCallback(BaseCallback):
    """Record finite reward components averaged across vector environments."""

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
                self.logger.record(f"reward/{key}", sum(samples) / len(samples))
        return True
