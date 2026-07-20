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

## Task 7: Real checkpoint and TensorBoard integration

### Installed API evidence

- `.venv\Scripts\python.exe` reported Stable-Baselines3 `2.9.0` and PyTorch
  `2.8.0+cpu`.
- `inspect.signature(PPO.__init__)` reported the positional `policy` and `env`
  parameters followed by PPO hyperparameters, including `n_steps: int = 2048`,
  `batch_size: int = 64`, `tensorboard_log: str | None = None`,
  `seed: int | None = None`, and `device: torch.device | str = 'auto'`.
- `inspect.signature(PPO.learn)` reported
  `(self, total_timesteps, callback=None, log_interval=1, tb_log_name='PPO',
  reset_num_timesteps=True, progress_bar=False)`.
- `inspect.signature(PPO.load)` reported
  `(path, env=None, device='auto', custom_objects=None, print_system_info=False,
  force_reset=True, **kwargs)`.
- `inspect.signature(CheckpointCallback.__init__)` reported
  `(self, save_freq, save_path, name_prefix='rl_model', save_replay_buffer=False,
  save_vecnormalize=False, verbose=0)`.
- `inspect.signature(EvalCallback.__init__)` reported
  `(self, eval_env, callback_on_new_best=None, callback_after_eval=None,
  n_eval_episodes=5, eval_freq=10000, log_path=None,
  best_model_save_path=None, deterministic=True, render=False, verbose=1,
  warn=True)`.

### RED attempt and correction decision

Command:

```powershell
.venv\Scripts\python.exe -m pytest tests/integration/test_ppo_checkpoint.py -v
```

The first real execution was unexpectedly GREEN: `1 passed, 15 warnings in 4.95s`.
No SB3 callback, path, constructor, learn, or load signature mismatch occurred. The
existing orchestration already supplied accepted SB3 2.9.0 argument names, passed its
callback list through `PPO.learn`, staged callback artifacts under the checkpoint
directory, and used the supported `PPO.load(..., env=..., device=...,
tensorboard_log=...)` `**kwargs` path for resume. Therefore no production correction
was justified and PPO defaults were not changed. The only observed runtime message was
SB3's advisory warning that the tiny evaluation environment was not wrapped in
`Monitor`; this test uses no reward- or episode-modifying wrapper, so artifact and
resume behavior are unaffected.

### GREEN evidence

The real integration uses a deterministic 24-element `float32` Box observation and
`Discrete(4)` action environment, with `n_steps=8`, `batch_size=8`,
`total_timesteps=16`, checkpoint/evaluation intervals of 8, and one evaluation episode.
It verifies periodic checkpoints at steps 8 and 16, `best_model.zip`,
`final_model.zip`, valid ZIP structure, a real `PPO.load` plus integer action prediction
in `[0, 3]`, a TensorBoard event file, and full equality between the serialized and
in-memory resolved configuration. It then resumes from the same run's canonical final
checkpoint for another 16 timesteps, loads and predicts from the promoted final model,
checks every promoted checkpoint is still a ZIP, confirms invocation staging is gone,
and confirms all four real train/evaluation environments were closed. All run artifacts
are confined to pytest's `tmp_path`.

Verification results:

- focused integration rerun: `1 passed, 15 warnings in 5.62s`;
- training/config regression: `41 passed, 14 warnings in 5.02s`;
- full regression: `420 passed, 15 warnings in 12.30s`;
- Ruff: `All checks passed!`;
- Ruff format: `78 files already formatted`;
- mypy: `Success: no issues found in 46 source files`;
- `git diff --check`: no whitespace errors (Git emitted only its Windows
  LF-to-CRLF working-copy notice for this Markdown file).
