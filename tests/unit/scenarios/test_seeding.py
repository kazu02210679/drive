import pytest

from mad_driving.config.models import SeedRangeConfig
from mad_driving.scenarios.seeding import EpisodeSeedAllocator


def test_allocator_is_reproducible_and_role_bounded() -> None:
    split = SeedRangeConfig(seed_start=10_000, seed_count=1_000)
    first = EpisodeSeedAllocator("validation", split, worker_index=2).allocate(42)
    second = EpisodeSeedAllocator("validation", split, worker_index=2).allocate(42)
    assert first == second
    assert first.episode_rng_seed == 42
    assert 10_000 <= first.metadrive_scenario_index < 11_000
    assert 10_000 <= first.scenario_parameter_seed < 11_000


def test_role_or_worker_changes_derived_seed_identity() -> None:
    split = SeedRangeConfig(seed_start=0, seed_count=10_000)
    assert (
        EpisodeSeedAllocator("train", split, worker_index=0).allocate(42)
        != EpisodeSeedAllocator("train", split, worker_index=1).allocate(42)
    )


def test_allocator_rejects_negative_worker_index() -> None:
    split = SeedRangeConfig(seed_start=0, seed_count=10)

    with pytest.raises(ValueError, match="worker_index"):
        EpisodeSeedAllocator("train", split, worker_index=-1)


def test_allocator_rejects_negative_episode_rng_seed() -> None:
    split = SeedRangeConfig(seed_start=0, seed_count=10)

    with pytest.raises(ValueError, match="episode_rng_seed"):
        EpisodeSeedAllocator("train", split, worker_index=0).allocate(-1)
