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
and confirms each parent-process training environment was closed. The process-isolated
evaluation environments are closed and joined by `SubprocVecEnv`; worker cleanup is
covered by the training lifecycle unit tests. All run artifacts are confined to
pytest's `tmp_path`.

Verification results:

- focused integration rerun: `1 passed, 15 warnings in 5.62s`;
- training/config regression: `41 passed, 14 warnings in 5.02s`;
- full regression: `420 passed, 15 warnings in 12.30s`;
- Ruff: `All checks passed!`;
- Ruff format: `78 files already formatted`;
- mypy: `Success: no issues found in 46 source files`;
- `git diff --check`: no whitespace errors (Git emitted only its Windows
  LF-to-CRLF working-copy notice for this Markdown file).

## Task 8: Canonical real smoke and Phase 4 verification

### Reproducible setup and exact versions

The documented install uses a normal, non-editable wheel and the training extra:

```powershell
.venv\Scripts\uv.exe sync --no-editable --group dev --extra training
```

The existing environment initially retained an older `mad-driving==0.1.0` wheel even
though `uv sync` reported it audited; `python -m mad_driving.cli.train` was therefore
absent. Rebuilding only the project wheel fixed the existing environment without
`PYTHONPATH`:

```powershell
.venv\Scripts\uv.exe sync --no-editable --group dev --extra training --reinstall-package mad-driving
```

The installed CLI and training module SHA-256 values then exactly matched their `src/`
files. Exact versions were Python `3.11.9`, uv `0.8.0`, mad-driving `0.1.0`, Gymnasium
`1.3.0`, MetaDrive `0.4.3`, NumPy `1.26.4`, Pydantic `2.11.7`, PyYAML `6.0.2`,
Stable-Baselines3 `2.9.0`, TensorBoard `2.19.0`, PyTorch `2.8.0` (CPU), pytest `8.4.1`,
pytest-cov `6.2.1`, Ruff `0.12.4`, and mypy `1.16.1`.

### Quality gate

The initial branch-range whitespace check found one committed extra blank line at EOF
in the Phase 4 design. That documentation-only defect was removed in the targeted
runtime-fix commit. The complete fresh gate after the correction was:

```powershell
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
git diff --check feat/phase3-control-shield...HEAD
```

Results: `421 passed, 19 warnings in 38.13s`; branch-aware total coverage `95.99%`
(`1949` statements, `56` missed, `396` branches, `38` partial); Ruff `All checks
passed!`; Ruff format `78 files already formatted`; mypy `Success: no issues found in
46 source files`; and the exact branch-range diff check exited 0 with no output.

### Canonical attempt, real defect, and minimal correction

`runs/phase4_smoke_seed42` did not exist before the canonical command. The non-editable
wheel was synchronized and the exact command was run without `PYTHONPATH`:

```powershell
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --smoke --run-dir runs/phase4_smoke_seed42
```

After 22.7 seconds it reached the step-5,000 evaluation boundary but exited 2 with
`training failed: Can not call this API after engine initialization!`. MetaDrive 0.4.3
supports at most one active `BaseEngine` per process; the single training `DummyVecEnv`
and separate evaluation `DummyVecEnv` attempted to initialize two engines in the parent
process. The failed directory was preserved and was not deleted or overwritten.

A focused real MetaDrive/PPO regression reproduced the exact assertion before the fix:
`1 failed, 15 warnings in 5.58s`. The minimal correction retains the documented
single-environment training `DummyVecEnv` but runs its evaluation environment in a
one-worker `SubprocVecEnv`, giving each real MetaDrive engine a separate process. The
same regression then passed: `1 passed, 16 warnings in 9.74s`; the focused training,
CLI, checkpoint, and MetaDrive regression set passed with `44 passed, 19 warnings in
19.22s`. The correction is commit `02ca645` (`fix: isolate metadrive evaluation
engine`). No PPO hyperparameter, reward, observation, scenario, or safety behavior
changed.

### Successful real smoke evidence

Because the first run directory now existed as failed evidence, the retry target was
checked absent and clearly named `runs/phase4_smoke_seed42_retry1`:

```powershell
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --smoke --run-dir runs/phase4_smoke_seed42_retry1
```

The headless CPU run exited 0 in `38.6s`. PPO requested `5,000` transitions and
completed `6,144` actual training transitions because `n_steps=2048` requires a whole
rollout. Evaluation ran at step `5,000`: five deterministic episodes of `200` steps
each, all ending by MetaDrive's `max_step` horizon truncation, with reward
`-24.707699465755887` for every episode. The successful verification therefore covered
`7,244` real simulator steps: `6,144` training + `1,000` evaluation + `100` checkpoint
reload steps. The reload itself did not terminate or truncate within its 100-step cap.

Artifacts:

```text
runs/phase4_smoke_seed42_retry1/
├── config_resolved.yaml                                      2,683 bytes
├── checkpoints/
│   ├── best_model.zip                                      169,749 bytes (step 5,000)
│   └── final_model.zip                                     169,749 bytes (step 6,144)
├── evaluation/
│   └── evaluations.npz                                         834 bytes
└── tensorboard/PPO_1/
    └── events.out.tfevents.1784560154.kaji.59412.0             3,938 bytes
```

The resolved YAML exactly records `smoke_timesteps: 5000`, `n_steps: 2048`, seed 42,
headless rendering, and the configured `checkpoint_interval_steps: 10000`.
`PPO.load` reported `num_timesteps=5000` for best and `6144` for final. No periodic
checkpoint was expected or produced because the successful run did not reach 10,000;
Task 7's real short-interval integration already proves periodic checkpoint save,
load, retention, and resume behavior.

### Fresh final-checkpoint prediction

The produced final checkpoint was loaded with `PPO.load(..., env=fresh_env,
device="cpu")` into a fresh `MultiAgentSpeedEnv` built from the resolved config. The
episode used `model.predict(observation, deterministic=True)`, checked every initial
and subsequent observation plus every reward with `numpy.isfinite`, stopped at
termination/truncation or 100 steps, and closed the environment in `finally`.

Result:

```json
{"checkpoint_num_timesteps": 6144, "finite_observations": true, "finite_rewards": true, "steps": 100, "terminated": false, "truncated": false, "last_metadrive_flags": {"arrive_dest": false, "crash_human": false, "crash_vehicle": false, "max_step": false, "out_of_road": false}}
```

### Warnings and phase boundary

- Existing Matplotlib/PyParsing deprecation warnings account for 14 collected warnings.
- SB3 advises that evaluation is not Monitor-wrapped. No reward- or episode-modifying
  wrapper is present, so recorded real lengths and rewards are unmodified.
- SB3 also advises that the single training and evaluation vector wrappers have
  different classes (`DummyVecEnv` and `SubprocVecEnv`). This is the intentional,
  tested process boundary required by MetaDrive's engine singleton.
- Git may warn that the user-global ignore file is sandbox-inaccessible or that LF will
  become CRLF in the Windows working copy; neither warning changes tracked content or
  gate results.
- Specialized scenarios and curriculum remain Phase 5. Evaluation artifacts,
  baselines, ablations, and visualization remain Phase 6. Task 8 Steps 5, 6, and 8
  remain controller-owned and were not performed here; no broad review, push, or PR was
  attempted.

## Task 8 review corrections and final smoke

The Task 8 review found three Important issues in the first successful smoke state:

1. the planned equal-count `DummyVecEnv` fallback for a failed multi-environment
   `SubprocVecEnv` would place multiple MetaDrive engines in one process;
2. normal `SubprocVecEnv.close()` had no bounded worker-exit verification; and
3. supplying `EvalCallback.log_path` produced `evaluations.npz`, although the approved
   Phase 4 boundary permits only resolved config, checkpoints, and TensorBoard events.

The fallback conflicted with the observed MetaDrive 0.4.3 process-wide engine singleton.
The design and plan were therefore amended to the minimum safe alternative required by
the top-level specification: partial subprocess resources are cleaned, worker exit is
verified, and construction fails explicitly instead of creating an unsafe same-process
fallback.

TDD evidence:

- RED: the new placement, teardown, exception-preservation, and artifact-boundary tests
  produced `6 failed, 27 passed`;
- GREEN: all training unit tests passed, `33 passed`;
- real/focused regression: training, CLI, real PPO checkpoint/resume, and real MetaDrive
  engine isolation passed, `47 passed, 19 warnings`;
- the real regression captures the evaluation `SubprocVecEnv` and asserts `closed=True`
  plus every worker process not alive after `run_training` returns.

Runtime teardown now sends the graceful close request when possible, uses bounded joins,
then escalates through terminate and kill, and confirms every worker stopped. A cleanup
failure replaces an otherwise successful result. If training already failed, that
original exception remains primary and receives a cleanup-failure note.

After reinstalling the non-editable project wheel, a fresh target was verified absent and
the canonical CLI was rerun:

```powershell
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --smoke --run-dir runs/phase4_smoke_seed42_postreview1
```

The command exited 0 in `37.1s`: `5,000` requested, `6,144` actual training
transitions, five evaluation episodes of `200` steps, best checkpoint step `5,000`, and
final checkpoint step `6,144`. The final artifact tree is:

```text
runs/phase4_smoke_seed42_postreview1/
├── config_resolved.yaml                                      2,683 bytes
├── checkpoints/
│   ├── best_model.zip                                      169,754 bytes (step 5,000)
│   └── final_model.zip                                     169,754 bytes (step 6,144)
└── tensorboard/PPO_1/
    └── events.out.tfevents.1784562285.kaji.47140.0             3,938 bytes
```

No `evaluation/` directory or `evaluations.npz` exists. A fresh
`MultiAgentSpeedEnv` then loaded the final checkpoint and completed 100 deterministic
prediction steps with finite observations and rewards, without termination or
truncation. Together the final verification exercised `7,244` simulator steps: `6,144`
training + `1,000` callback evaluation + `100` checkpoint reload.

### Broad-review lifecycle follow-up

The first whole-branch re-review found no Critical issue and three further Important
lifecycle gaps: staging-directory deletion could mask a training exception, an
intermediate process-operation exception could skip the later kill stage, and failed
partial-vector cleanup could replace the original constructor exception. It also found
one minor config mismatch: `standstill_speed_mps` allowed zero although speed thresholds
are positive in the approved design.

Additional TDD evidence:

- RED: `6 failed, 39 passed` for staging cleanup, operation escalation, constructor
  exception preservation, and the zero threshold;
- GREEN: the focused config/training unit set passed, `45 passed, 14 warnings`;
- real/focused regression passed, `59 passed, 19 warnings`, including real PPO
  save/load/resume and real MetaDrive subprocess isolation.

Each process operation is now independently guarded, so join or terminate failure does
not prevent the final kill attempt. Final liveness remains the success criterion.
Staging and vector cleanup failures are aggregated: they replace an otherwise successful
return, but are attached as notes when a primary training or construction exception
already exists. `standstill_speed_mps` now validates with `gt=0.0`.
