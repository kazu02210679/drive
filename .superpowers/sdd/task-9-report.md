# Phase 4.1 Task 9 Report: Role-aware environment factories

## Status and scope

Implemented Task 9 on `feat/phase4-rl-environment` from base `43081d7`.
The change is limited to role-aware training environment ownership, strict actual
MetaDrive scenario identity, and focused unit/real-integration coverage. No Task 10
run ownership or resume provenance, Task 11 documentation migration, Phase 5 scenarios,
or Phase 6 evaluation behavior was added. Nothing was pushed.

No governing `AGENTS.md` exists in the repository or checked parent directories. The
Phase 4.1 design and implementation plan, Task 9 brief, progress ledger, Task 1 seed
configuration report, and Task 8 environment lifecycle report were read completely
before implementation.

## Implementation

- Changed the training `EnvironmentFactory` protocol to
  `factory(config, *, role, worker_index)` with the public `EnvironmentRole` type.
- Training now constructs exactly `num_envs` train environments with worker indices
  `0..num_envs-1`, followed by exactly one validation environment at worker index `0`
  for `EvalCallback`. Training never requests the test role.
- Each vector-environment thunk binds its role and worker index with `partial` and owns
  a deep `AppConfig` copy. Cloudpickle round trips preserve every identity and do not
  share config objects across workers.
- The default factory forwards the explicit identity to `MultiAgentSpeedEnv`. Its
  existing Task 8 boundary creates a fresh MetaDrive dictionary for each environment
  and overrides `start_seed`/`num_scenarios` from the selected `AppConfig.scenarios`
  split without mutating the shared source config.
- Removed modulo wrapping from `ControlMetaDriveEnv.reset`. Explicit scenario indices
  must be integers inside the configured half-open range before `MetaDriveEnv.reset`
  is called.
- The control adapter copies reset info, reconciles MetaDrive's `env_seed` with
  `current_seed`, verifies an explicit request exactly matches the returned identity,
  and publishes `metadrive_scenario_index` as the actual simulator value.
- Migrated the real PPO integration factory to the new explicit role/worker contract
  and independent-config expectation.
- Added real validation coverage proving actual indices remain in `[10000, 11000)` and
  match both reset info and the simulator's `current_seed`.
- Added a real MetaDrive end-to-end emergency test. A deterministic integration suite
  creates a hard-stop decision after a warm-up step; enforce mode sends STOP through
  `MultiAgentSpeedEnv` to the live `LaneKeepingLongitudinalPolicy`, produces target
  speed `0`, full emergency brake, and resets speed-PID state. Monitor mode reports the
  same required STOP while sending the unchanged requested KEEP action to the policy.

The Gymnasium/SB3 APIs, `(24,)` `float32` observation, four-action ordering, Task 8 seed
progression, and role-disjoint allocator derivation are unchanged.

## Strict TDD evidence

### RED

The role/factory, cloudpickle identity, independent-config, invalid identity, and real
out-of-range reset tests were written before production edits.

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/training/test_train.py::test_training_constructs_train_workers_and_validation_eval tests/unit/training/test_train.py::test_environment_thunks_capture_independent_config_copies tests/unit/training/test_train.py::test_default_environment_factory_rejects_invalid_identity tests/unit/training/test_train.py::test_cloudpickle_round_trip_preserves_each_role_and_worker_without_shared_config tests/integration/test_rl_metadrive_headless.py::test_real_control_adapter_rejects_out_of_range_before_simulator_side_effects -q
```

Result: `7 failed`. The failures were the expected missing keyword-only factory inputs,
the old one-argument owner call, and the real adapter accepting seed `39` by modulo
wrapping and starting MetaDrive instead of raising before side effects.

The emergency integration was also added before production edits. Its enforce and
monitor paths exercise existing Task 8/Phase 3 behavior, so no policy implementation
change was required. After changing the warm-up request from KEEP to SLOW to avoid an
unrelated safe speed-cap diagnostic, both real paths passed.

### GREEN

The same focused role/reset command passed `7 passed, 14 warnings` after the minimal
factory and adapter implementation.

The complete training unit file passed:

```text
44 passed, 14 warnings
```

The complete Task 9 real MetaDrive integration set passed:

```powershell
.venv\Scripts\python.exe -m pytest tests/integration/test_control_metadrive_headless.py tests/integration/test_rl_metadrive_headless.py -q
```

```text
14 passed, 16 warnings
```

These tests are not skipped and construct real MetaDrive 0.4.3 environments.

## Verification

Untouched baseline before tests or production changes:

```text
513 passed, 19 warnings in 33.68s
```

Focused static checks after formatting only Task 9 files:

```text
Ruff check on envs, training, training unit tests, and integrations: All checks passed!
Ruff format --check on six touched code/test files: 6 files already formatted
mypy src/mad_driving/envs src/mad_driving/training:
Success: no issues found in 7 source files
git diff --check: no whitespace errors
```

Full repository suite after implementation:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

```text
522 passed, 19 warnings in 33.80s
```

Warnings remain the pre-existing third-party Matplotlib/PyParsing deprecations and
Stable-Baselines3 evaluation-wrapper warnings.

## Files changed

- `src/mad_driving/envs/control_metadrive_env.py`
- `src/mad_driving/training/train.py`
- `tests/unit/training/test_train.py`
- `tests/integration/test_control_metadrive_headless.py`
- `tests/integration/test_rl_metadrive_headless.py`
- `tests/integration/test_ppo_checkpoint.py`
- `.superpowers/sdd/task-9-report.md`

## Concerns

None. The real emergency-path assertion intentionally inspects MetaDrive's live policy
registry and controller diagnostics because normalized brake output alone cannot prove
that Shield-enforced STOP reached the policy's emergency branch.
