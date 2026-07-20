# Task 2 report: reproducible role-scoped seed identities

## Implementation

- Added `EnvironmentRole = Literal["train", "validation", "test"]`.
- Added frozen `EpisodeSeeds` with the episode RNG seed, MetaDrive scenario index, and scenario-parameter seed.
- Added frozen `EpisodeSeedAllocator`, consuming `SeedRangeConfig` and deriving two child sequences from NumPy `SeedSequence([episode_rng_seed, role_code, worker_index])`.
- Each child emits one `uint32`, which is reduced modulo `seed_count` and offset by `seed_start`.
- Added explicit `ValueError` validation for negative worker and episode RNG seeds.
- Added the scenarios package initializer.

## Files

- `src/mad_driving/scenarios/__init__.py`
- `src/mad_driving/scenarios/seeding.py`
- `tests/unit/scenarios/test_seeding.py`

## TDD evidence

1. Wrote the two deterministic allocation tests before creating the scenarios production package.
2. RED: `.venv\\Scripts\\python.exe -m pytest tests/unit/scenarios/test_seeding.py -q` failed during collection with `ModuleNotFoundError: No module named 'mad_driving.scenarios'`.
3. Implemented the minimum allocator and ran the same focused command: GREEN, `2 passed in 0.18s`.
4. Added tests for the required negative worker and episode seed validation, temporarily leaving validation absent.
5. RED: the focused command reported `2 failed, 2 passed`; worker construction did not raise, and NumPy raised `ValueError: expected non-negative integer` rather than the required `episode_rng_seed` validation message.
6. Added explicit validation and reran the focused command: GREEN, `4 passed in 0.17s`.

## Verification

- Focused pytest: `4 passed`.
- Ruff: `.venv\\Scripts\\ruff.exe check src/mad_driving/scenarios tests/unit/scenarios` — `All checks passed!`
- Mypy: `.venv\\Scripts\\mypy.exe src/mad_driving/scenarios` — `Success: no issues found in 2 source files`.
- Full pytest: `438 passed, 19 warnings in 26.79s`.

## Self-review

- The allocator is deterministic for identical role, range, worker, and episode inputs.
- Role codes and worker indices are included in the entropy, while both derived identities remain inside the configured half-open range.
- Both result and allocator dataclasses are frozen, and no environment integration was added.
- The focused tests cover reproducibility, bounds, worker identity separation, and required negative-input validation.

## Concerns

- The full suite emitted 19 existing dependency/runtime warnings from matplotlib and Stable-Baselines3. They did not affect test outcomes and are unrelated to the new scenarios package.
