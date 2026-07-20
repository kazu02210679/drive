# Phase 4 Implementation Log

## Task 5: Real MetaDrive Gymnasium compliance

### Installed API evidence

- `.venv\Scripts\python.exe` reported Gymnasium `1.3.0` and
  `metadrive-simulator` `0.4.3` through the installed package metadata.
- Gymnasium `env_checker.check_reset_seed_determinism()` calls `reset(seed=123)` and
  `reset(seed=456)` as arbitrary RNG seeds.
- MetaDrive `BaseEnv.reset()` documents its `seed` as the scenario index. Its
  `_reset_global_seed()` asserts that the value is in
  `[start_index, start_index + num_scenarios)`.
- MetaDrive `TerminationState` defines the exact keys `arrive_dest`, `crash_vehicle`,
  `crash_human`, `out_of_road`, and `max_step`. A real reset at scenario index 42 and
  one headless step returned all five keys with finite API output:
  `{'arrive_dest': False, 'crash_vehicle': False, 'crash_human': False,
  'out_of_road': False, 'max_step': False}`.

### RED

Command:

```powershell
.venv\Scripts\python.exe -m pytest tests/integration/test_rl_metadrive_headless.py -v
```

Result: `1 failed, 2 passed`. The first concrete mismatch was
`test_real_rl_environment_passes_gymnasium_checker`: Gymnasium called
`reset(seed=123)`, and the wrapper forwarded 123 directly to MetaDrive. With
`configs/base.yaml` defining `start_seed: 42` and `num_scenarios: 1`, MetaDrive raised
`AssertionError: scenario_index (seed) should be in [42:43)`.

### Cause and minimal correction

The concrete MetaDrive binding treated Gymnasium's arbitrary RNG seed and MetaDrive's
bounded scenario index as the same value. `MultiAgentSpeedEnv.reset()` still forwards
the caller's seed unchanged to its simulator factory. `ControlMetaDriveEnv.reset()` now
normalizes that value into the configured scenario range with the same offset-modulo
formula already used by MetaDrive's `BaseEngine.seed()`. For `base.yaml`'s sole scenario,
every arbitrary Gymnasium seed maps to scenario index 42. No scenario, reward rule,
termination rule, generic simulator-factory contract, or info-key mapping changed.
An initial wrapper-level adaptation was discarded after the relevant unit regression
gate reported `20 failed, 93 passed`; the final concrete-binding correction restored
that suite to `113 passed`.

### GREEN

The same focused command completed with `3 passed, 14 upstream warnings in 6.17s`.
The real tests cover Gymnasium `check_env`, a deterministic 100-step action sequence
with finite 24-element `float32` observations/rewards/components and the exact
MetaDrive info keys, and two seed-42 runs with identical initial observations and first
10 decision traces. Every real environment is closed in a `finally` block.

Relevant unit regressions completed with `113 passed`; Ruff reported `All checks
passed!`, Ruff format reported `69 files already formatted`, mypy reported no issues in
42 source files, and `git diff --check` reported no whitespace errors. The final full
project suite completed with `376 passed, 14 upstream warnings in 8.32s`.
