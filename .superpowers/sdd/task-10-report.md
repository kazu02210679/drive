# Task 10 Report: Strict Run Ownership and Resume Provenance

## Outcome

Implemented Phase 4.1 Task 10 only. New and resumed training now require an absent or
empty destination, resume reads a separate version-2 source run without modifying it,
and every fresh/continued run writes complete versioned provenance metadata.

## RED / GREEN evidence

- Baseline: `522 passed, 19 warnings`.
- Ownership RED: the behavioral run produced `2 failed, 1 passed`; non-empty directories
  were still reused and the CLI still called training. The file case already had the
  required error.
- Ownership GREEN: `3 passed` with exact errors and preserved bytes/listings.
- Metadata API RED: collection failed because `RunMetadata`, `ResumeMetadata`, and
  `sha256_file` did not exist.
- Fresh metadata GREEN: `3 passed`.
- Resume provenance RED: `21 failed`; metadata was ignored, incompatible loaded PPO
  fields/spaces learned, source changes were not detected, and resume metadata was absent.
- Resume provenance GREEN: all `21 passed` before legacy same-run tests were migrated.
- Final focused training/CLI/real-checkpoint suite: `93 passed, 17 warnings`.

## Implementation

- Added frozen `RunMetadata` and `ResumeMetadata` public models and public
  `sha256_file` / `validate_resume_contract` exports.
- Metadata records research contract 2, observation schema 1 and `(24,)`, action schema 1,
  action count 4, and exact `KEEP/SLOW/PREPARE_STOP/STOP` order.
- Destination ownership is checked before environment/model/log/artifact side effects and
  rechecked immediately before artifact creation. Exact file/non-empty errors are retained.
- Resume source discovery resolves the nearest source run metadata and resolved config,
  rejects missing, malformed, legacy, schema-incompatible, or inconsistent sources, and
  prohibits destinations inside the source run.
- Parent checkpoint SHA-256 is computed before load and recomputed after `PPO.load`; a
  change aborts before destination writes or learning.
- Post-load validation compares policy, numeric/effective learning-rate representation,
  `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, effective `clip_range`,
  `ent_coef`, `vf_coef`, `max_grad_norm`, exact Box observation space, and Discrete action
  space before `learn`.
- Resume metadata records canonical parent checkpoint/run paths, lowercase SHA-256,
  source config, deterministic allowed-field config diff, and loaded
  `start_num_timesteps`; resumed learning uses `reset_num_timesteps=False`.
- Metadata uses a sibling temporary file plus `os.replace`; temporary cleanup preserves
  the primary serialization/replacement error.
- Migrated real PPO checkpoint coverage to a separate continuation destination and proved
  the complete source run remains byte-for-byte unchanged.

## Files

- `src/mad_driving/training/metadata.py` (new)
- `src/mad_driving/training/train.py`
- `src/mad_driving/training/__init__.py`
- `src/mad_driving/cli/train.py`
- `tests/unit/training/test_metadata.py` (new)
- `tests/unit/training/test_train.py`
- `tests/unit/cli/test_train.py`
- `tests/integration/test_ppo_checkpoint.py`
- `.superpowers/sdd/task-10-report.md` (new)

## Verification

- Full pytest: `558 passed, 19 warnings`.
- Ruff: all checks passed.
- mypy strict: no issues in 52 source files.
- `git diff --check`: clean; Git emitted only expected Windows LF-to-CRLF notices.

## Concerns

- The checkpoint TOCTOU defense detects content changes spanning validation/load by hashing
  before and immediately after `PPO.load`. It does not lock the source file; a hostile actor
  that changes and restores identical bytes entirely between those reads is outside this
  filesystem-level contract.
- The 19 warnings are unchanged upstream Matplotlib/SB3 warnings; no Task 11 documentation,
  smoke execution, push, Phase 5, or later-phase work was performed.

---

## Independent-review remediation

The Task 10 independent review returned **CHANGES REQUIRED**. The focused follow-up fixes
all five findings without entering Task 11.

### Additional RED / GREEN evidence

- Atomic ownership RED: the three destination-race tests failed because preflight and the
  first directory write were separated. GREEN: all three passed after adding an atomic
  directory/exclusive-marker claim held through artifact publication.
- Marker cleanup RED: both absent and pre-existing-empty cases left `.training-owner`
  behind when marker initialization failed. GREEN: `5 passed` across acquisition races,
  two compliant concurrent writers, and both marker-write failure cases.
- Metadata RED: `26 failed, 3 passed` for recursive immutability, exact public-constructor
  validation, observation dtype, canonical paths/digests/timesteps, and finite-only JSON.
  GREEN: `29 passed`.
- Schedule/timestep RED: the old three-point sampler accepted a callable that differed at
  progress `0.25`; bool and fractional `num_timesteps` were silently converted. The
  adversarial subset exposed `5 failed, 4 passed`. GREEN: `9 passed` after structural SB3
  schedule validation and raw integral validation.
- Final focused training/CLI/checkpoint gate: `135 passed, 17 warnings`, including the real
  PPO checkpoint continuation integration.

### Remediation implementation

- Added a Windows-compatible `O_CREAT | O_EXCL` ownership marker. Absent destinations use
  atomic `mkdir`; pre-existing empty destinations claim the marker and are rechecked for
  foreign entries. The descriptor remains open until final publication and cleanup.
- Ownership cleanup verifies the still-linked marker by file identity before unlinking it,
  preserves competitor files/markers, removes only a process-created empty directory, and
  adds cleanup failures as notes without replacing the primary exception.
- `RunMetadata` and `ResumeMetadata` now validate every public field and recursively detach
  JSON mappings/sequences into deterministic immutable values. Contract/schema/action
  identity, `float32` observation dtype, canonical paths, lowercase SHA-256, required
  mappings, and non-bool nonnegative integral timesteps are enforced at construction.
- Metadata loading rejects `NaN`, `Infinity`, and `-Infinity` through `parse_constant` and
  recursively validates nested values. Writing validates before temporary-file creation,
  uses `allow_nan=False`, and retains sibling-temp cleanup plus atomic `os.replace`.
- Effective learning-rate and clip-range schedules must have the exact pinned SB3 2.9
  `FloatSchedule(ConstantSchedule(value))` structure and state. Arbitrary callables are no
  longer sampled or accepted.
- Loaded `model.num_timesteps` is checked as a non-bool, nonnegative `Integral` before any
  conversion, metadata write, or learning.
- The frozen `seed_compatible_source_run` fixture now writes realistic version-2 metadata
  through the production transactional serializer. Adversarial source tests confirm exact
  byte preservation and untouched failed destinations.

### Final verification after remediation

- Full pytest: `597 passed, 19 warnings`.
- Ruff lint: `All checks passed!` across `src` and `tests`.
- Ruff format: all 5 touched Python files already formatted.
- mypy strict: `Success: no issues found in 53 source files`.
- `git diff --check`: clean apart from expected Windows LF-to-CRLF notices.

### Remaining concern

- Source-checkpoint TOCTOU handling remains detection rather than locking: SHA-256 is
  checked before and immediately after `PPO.load`. A hostile writer that changes and then
  restores byte-identical content entirely between those reads is outside this contract.
  Destination ownership itself is exclusive and held for the complete artifact transaction.
