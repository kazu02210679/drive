"""Deterministic per-episode scenario parameter sampling."""

from numbers import Integral
from typing import TypeVar

import numpy as np
from numpy.random import Generator

_Choice = TypeVar("_Choice")


class ScenarioParameterSampler:
    """Own one local generator seeded only by scenario_parameter_seed."""

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("scenario parameter seed must be an integer")
        self._generator: Generator = np.random.default_rng(int(seed))

    def uniform(self, name: str, minimum: float, maximum: float) -> float:
        """Sample one named continuous parameter from its validated range."""

        del name
        return float(self._generator.uniform(minimum, maximum))

    def choose(self, values: tuple[_Choice, ...]) -> _Choice:
        """Choose one item from a stable, non-empty ordered scenario table."""

        if not values:
            raise ValueError("scenario choices must not be empty")
        return values[int(self._generator.integers(0, len(values)))]
