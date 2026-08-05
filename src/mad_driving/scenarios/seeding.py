"""Deterministic, role-scoped episode seed allocation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from mad_driving.config.models import SeedRangeConfig

EnvironmentRole = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class EpisodeSeeds:
    """Seeds and scenario identity allocated for one episode."""

    episode_rng_seed: int
    metadrive_scenario_index: int
    scenario_selection_seed: int
    scenario_parameter_seed: int


_ROLE_CODES: dict[EnvironmentRole, int] = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


@dataclass(frozen=True)
class EpisodeSeedAllocator:
    """Derive reproducible scenario identities within one role's seed range."""

    role: EnvironmentRole
    seed_range: SeedRangeConfig
    worker_index: int

    def __post_init__(self) -> None:
        if self.worker_index < 0:
            raise ValueError("worker_index must be non-negative")

    def allocate(self, episode_rng_seed: int) -> EpisodeSeeds:
        """Allocate bounded scenario seeds for an episode RNG seed."""

        if episode_rng_seed < 0:
            raise ValueError("episode_rng_seed must be non-negative")
        sequence = np.random.SeedSequence(
            [episode_rng_seed, _ROLE_CODES[self.role], self.worker_index]
        )
        road, selection, parameters = sequence.spawn(3)
        return EpisodeSeeds(
            episode_rng_seed=episode_rng_seed,
            metadrive_scenario_index=self._bounded(road),
            scenario_selection_seed=self._bounded(selection),
            scenario_parameter_seed=self._bounded(parameters),
        )

    def _bounded(self, child: np.random.SeedSequence) -> int:
        value = child.generate_state(1, dtype=np.uint32)[0]
        return self.seed_range.seed_start + int(value) % self.seed_range.seed_count
