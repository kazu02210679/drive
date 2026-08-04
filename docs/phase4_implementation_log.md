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

A targeted re-review then found that a failed parent-pipe `remote.close()` during partial
construction cleanup was not included in that note. A RED regression reproduced the
missing note; the cleanup now reports either unconfirmed worker exit or remote closure
while preserving the constructor exception. The final training unit set passed,
`38 passed, 14 warnings`.

### Final review and quality gate

The final targeted review reported no Critical, Important, or Minor issues and assessed
the branch as `Ready to merge: Yes`. The complete Phase 4 gate was then rerun from the
reviewed branch head:

```powershell
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
git diff --check feat/phase3-control-shield...HEAD
```

Results: `430 passed, 19 warnings in 33.35s`; total branch-aware coverage `95.76%`
(`2002` statements, `57` missed, `428` branches, `46` partial); Ruff `All checks
passed!`; Ruff format `78 files already formatted`; mypy found no issues in 46 source
files; and the branch-range whitespace check exited 0 with no output.

The branch was pushed as `feat/phase4-rl-environment`, and stacked Draft PR
[#4](https://github.com/kazu02210679/drive/pull/4) was opened against
`feat/phase3-control-shield`. It explicitly depends on Phase 3 Draft PR #3.

## Phase 4.1 Task 11: research-hardening verification

### Standalone environment resolution

The checkout uses a normal non-editable wheel. Before verification, the installed `mad_driving.cli.train` SHA-256 differed from `src/mad_driving/cli/train.py`. The checkout-local environment was refreshed without changing global Python:

```powershell
.venv\Scripts\uv.exe sync --no-editable --group dev --extra training --reinstall-package mad-driving
```

`uv 0.8.0` rebuilt and reinstalled `mad-driving==0.1.0`. The installed and source CLI SHA-256 then both equaled `758ac5034baccc02f0ec4b14f86e7d9f4f200a0ef4f42447399c55846e4fcb3b`, so standalone `python -m mad_driving.cli.train` resolved this checkout.

### Fresh quality and real MetaDrive gates

The final Task 11 gate used the requested commands:

- `pytest -q`: `608 passed, 19 warnings in 36.37s`
- `ruff check .`: `All checks passed!`
- `mypy src`: `Success: no issues found in 53 source files`
- coverage pytest with `--cov-fail-under=90`: `608 passed, 19 warnings in 50.03s`; `3,201` statements, `227` missed, total `90.42%`
- real MetaDrive trio: `15 passed, 16 warnings in 18.54s`

The 19 full-suite warnings comprise 14 upstream Matplotlib/PyParsing deprecations and five Stable-Baselines3 advisories: three train/eval vector-class warnings and two unwrapped-evaluation warnings. The three-file MetaDrive run contains the 14 deprecations plus two Stable-Baselines3 advisories. No project warning was suppressed.

`Get-Process python,pythonw` reported `count=0` before the fresh gates and `count=0` after the real integration trio. It again reported `count=0` after both training runs and independent artifact reloads. No Python/MetaDrive process attributable to verification remained.

### Reproducible real PPO smoke pair

Read-only `Test-Path` checks reported both planned destinations absent. No run or recovery directory was removed or overwritten:

```text
runs\phase4_1_smoke_seed42_a  exists=False
runs\phase4_1_smoke_seed42_b  exists=False
```

Both runs used `configs/train.yaml`, `--smoke`, policy/RNG seed 42, and fresh destinations. A completed in 43.1s and B in 42.1s. Each requested 5,000 timesteps, completed 6,144 at the full `n_steps=2048` rollout boundary, selected a best checkpoint at step 5,000, and evaluated five 200-step episodes with mean reward `-64.71`.

An independent parser loaded each resolved YAML, strict JSON metadata, TensorBoard event file, best/final checkpoint, and fresh real MetaDrive environment. The first three reproduced identities were identical:

| Role | `episode_rng_seed` | `metadrive_scenario_index` | `scenario_parameter_seed` |
|---|---:|---:|---:|
| train | 42 | 948 | 2,314 |
| train | 191,664,963 | 3,546 | 2,463 |
| train | 1,662,057,957 | 5,299 | 6,974 |
| validation | 42 | 10,746 | 10,418 |
| validation | 191,664,963 | 10,103 | 10,696 |
| validation | 1,662,057,957 | 10,595 | 10,383 |

Each train sequence contains three distinct episode RNG seeds, and A/B sequences match exactly. Every train scenario/parameter identity is in `[0, 10000)`. Every validation identity is in `[10000, 11000)` and therefore outside the train range. Test seeds were not constructed or consumed.

Both metadata files are byte-identical with SHA-256 `e8788a5a76c03b67d9021dabdff8614fefda1ea5d4dc75b70f51edd34da4beaa`; both resolved configs are byte-identical with SHA-256 `682942f33278ee5e515129b717a984afd4ca3bb9442eab06a062e99f12fe66d6`. Each run records `research_contract_version=2`, `observation_schema_version=1`, shape `[24]`, dtype `float32`, action schema 1, action count 4, and `resume=null`.

Each TensorBoard run contains one event file with 63 scalar events across evaluation, ten Reward components, timing, and PPO training tags; every parsed scalar is finite. Each final checkpoint reloaded with `num_timesteps=6144` into a fresh train-role environment and completed 100 deterministic prediction steps with finite observations and rewards, without termination or truncation. The differing checkpoint ZIP hashes are artifact-container bytes and were not used as the reproducibility criterion; config, metadata, seed identity sequences, timestep counts, and parsed finite behavior match.

### Deferred observation work and phase boundary

The current Coordinator observation remains 24-dimensional. Explicit `ttc_valid`, `claim_valid`, `agent_failed`, and `target_actor_present` features are deferred and unimplemented; no documentation claims that these slots exist. Adding them requires a versioned observation-schema change and retraining.

Phase 4.1 adds no Phase 5 Lead Brake, Cut-in, or Occluded Crossing Actor and no Curriculum. It adds no Phase 6 baseline execution, reports, plots, or GIFs. Task 11 changed documentation/evidence only and did not push; the parent performs the final whole-branch review and PR #4 push.

## Phase 4.1 Task 11 review remediation: actual reset seed artifacts

The original Task 11 smoke evidence above regenerated seed sequences after training. It did not prove which reset identities the PPO train and validation environments actually consumed. That seed-table evidence is superseded. The old run directories remain preserved and were not modified or deleted.

Training now wraps every train and validation environment at the actual Gymnasium reset boundary. Each initial reset and VecEnv auto-reset durably appends strict finite JSON to a private-workspace file named `episode_seeds/<role>-worker-<index>.jsonl`. Per-role/worker files avoid writer collisions for `num_envs>1`. Invalid or incomplete seed info raises without writing a record. The runner closes all VecEnv workers before validating file identity, count, schema, and SHA-256, rewrites `run_metadata.json` with `episode_seed_artifacts`, and only then atomically publishes the private workspace. The research contract remains version 2; version-2 resume metadata that predates the optional inventory loads with an empty artifact tuple.

Strict TDD began with `ModuleNotFoundError: No module named 'mad_driving.training.episode_seeds'`. After the first integration implementation, three orchestration tests still failed because cleanup assertions had to inspect the wrapped environment and one failed worker close was retried. The final focused training set passed `144 passed, 17 warnings`. Added coverage-branch tests brought the final full count to 628. No production change was made after the implementation became GREEN.

The checkout-local non-editable package was refreshed without changing global Python:

```powershell
.venv\Scripts\uv.exe sync --no-editable --group dev --extra training --reinstall-package mad-driving
```

`uv 0.8.0` rebuilt and reinstalled `mad-driving==0.1.0`. Source and installed `episode_seeds.py` both had SHA-256 `1fa57f56c98f7ed2fd3dc6e8194a06988ab34e2a6b8dcfb72eb9721561495e00`.

### Fresh remediation gates

- `.venv\Scripts\python.exe -m pytest -q`: `628 passed, 19 warnings in 38.67s`
- `.venv\Scripts\ruff.exe check .`: `All checks passed!`
- `.venv\Scripts\mypy.exe src`: `Success: no issues found in 54 source files`
- coverage with `--cov-fail-under=90`: `628 passed, 19 warnings in 50.48s`; 3,368 statements, 244 missed, 902 branches, 147 partial branches, exact total `90.19%`
- real MetaDrive trio: `15 passed, 16 warnings in 16.64s`

The warning inventory remains 14 upstream Matplotlib/PyParsing deprecations plus five Stable-Baselines3 advisories in the full runs, and the same 14 deprecations plus two Stable-Baselines3 advisories in the real trio. No warning was suppressed. A first direct `pytest.exe -q` attempt stopped during collection with 12 `ModuleNotFoundError: tests` errors because that launcher omitted the checkout root; the canonical `python -m pytest` invocation above passed. The first coverage attempt ran all 622 then-current tests but failed the strict threshold at `89.91%`; added artifact-boundary tests produced the passing `90.19%` result.

Process checks reported `python_process_count=0` before verification, after the real trio, and after both smoke reloads. No Python or MetaDrive process attributable to the gates remained.

### Fresh persisted-artifact smoke pair

Read-only checks confirmed both destinations absent before use:

```text
runs/phase4_1_seed_artifact_smoke_20260721_a exists=False
runs/phase4_1_seed_artifact_smoke_20260721_b exists=False
```

Both standalone CLI commands used `configs/train.yaml`, smoke mode, and seed 42. Run A completed in `40.562s`; run B completed in `40.217s`. Each requested 5,000 steps, completed 6,144 at the PPO rollout boundary, selected the best checkpoint at 5,000 steps, and evaluated five 200-step episodes at mean reward `-64.71`.

The parser read only the newly persisted artifacts. Both runs contain 31 train records and 6 validation records; the complete train and validation triple sequences are byte-identical across A and B. The shared train `episode_rng_seed` sequence is:

```text
[42, 191664963, 1662057957, 1405681631, 942484272, 929893137,
 1843824992, 184566854, 1497586438, 432652533, 202244314, 1130604997,
 2095133045, 1580016183, 1634535062, 1540770719, 1688060240,
 1102145672, 275121930, 1803345590, 967196436, 1074497555, 796282693,
 392022359, 1990212658, 1678403330, 1382689815, 864178266, 1766867109,
 1171300112, 952224740]
```

The shared validation sequence contains six triples:

```text
(42, 10746, 10418)
(191664963, 10103, 10696)
(1662057957, 10595, 10383)
(1405681631, 10643, 10445)
(942484272, 10361, 10833)
(929893137, 10640, 10594)
```

The first three train triples are `(42, 948, 2314)`, `(191664963, 3546, 2463)`, and `(1662057957, 5299, 6974)`. Every train scenario and parameter identity is in `[0, 10000)`; every validation scenario and parameter identity is in `[10000, 11000)`, disjoint from train. All records contain the three integer identities, role, and worker index. The train JSONL SHA-256 is `a27983d91ae73cb406209fe8ef7d17e3cbaf98912728249cad90321f06f35093`; validation is `cf2c51fef0bb731a6c579f4a06964ba67608fc18d2e6093c91a5777e48dcd445`. Each value matches its metadata descriptor.

Both metadata files are byte-identical at SHA-256 `37dfdb2451cc2032ed4d56c3e2f65dfdb3381dd2c763ec2ff45f14d7fc7a2489`; configs remain byte-identical at `682942f33278ee5e515129b717a984afd4ca3bb9442eab06a062e99f12fe66d6`. Each metadata file records research contract 2, observation schema 1, shape `[24]`, dtype `float32`, action schema 1, `resume=null`, and seed artifact schema 1. Each TensorBoard event file contains 63 finite scalar values.

The final checkpoint SHA-256 values are `191b77cf7d471eb8850ffc1e130c427a0e46f194e258c892d6ef1a1ac761a4d4` for A and `d4b1c487e15730d8187889193ed325fb018808d7ee6560cb674d8b64bf444f19` for B. Each reloaded at `num_timesteps=6144` and completed 100 deterministic real MetaDrive decision steps with 101 finite observations, 100 finite rewards, no termination, and no truncation. Checkpoint ZIP byte equality is not the deterministic training contract.

The Coordinator Observation remains 24-dimensional. `ttc_valid`, `claim_valid`, `agent_failed`, and `target_actor_present` remain deferred and unimplemented. No Phase 5 Actor/Curriculum or Phase 6 work was added. Nothing was pushed.

## 2026-07-21 Phase 4.1 final whole-branch fix wave

This final section supersedes all earlier Phase 4.1 verification and smoke evidence in this
log. Historical runs remain preserved; no run or recovery directory was deleted or reused.

Strict RED→GREEN tests closed the complete-transition timing race, typed privileged
termination boundary, descriptor/path seed-artifact replacement races, cleanup exception
semantics, genuine no-spawn fallback, required CLI destination, and cross-host chained
resume provenance. Timing is checked after the simulator step and after the runtime hook
immediately before next-frame construction. Off-road is terminating; unmatched raw
termination is a consistency error. Seed schema 2 binds every writer and parent inventory
to platform file identity and one parse/hash byte read. Cleanup failure prevents atomic
publication unless it is attached as a note to an existing primary exception.

For `num_envs=1`, train and validation now use sequential `DummyVecEnv` instances. Training
closes before one deferred validation pass, satisfying MetaDrive's process-global engine
constraint without invoking `SubprocVecEnv`. For `num_envs>1`, train remains subprocessed
and validation worker 0 remains a parent-process `DummyVecEnv`. `--run-dir` is required
before config loading, and historical parent paths are validated as provenance strings
without current-host path parsing. Observation remains exactly 24-dimensional; Phase 5 and
Phase 6 were not expanded.

### Superseding gates

- canonical full pytest: `655 passed, 16 warnings in 27.11s`;
- Ruff lint: `All checks passed!`;
- mypy: `Success: no issues found in 54 source files`;
- strict coverage: `655 passed, 16 warnings in 33.72s`, 3,535 statements, 256 missed,
  960 branches, 159 partial, exact `90.28%`;
- real MetaDrive trio: `15 passed, 15 warnings in 13.80s`.

The first strict coverage attempt after implementation ran 642 tests and failed at
`89.69%`. Focused seed descriptor/parser boundaries plus the final one-shot logger and
duplicate-key tests raised the suite to 655 and the passing `90.28%`; the failed result is
not reported as a completion gate. No warning was suppressed.

### Superseding fresh PPO smokes

Read-only preflight checks proved both destinations absent:

```text
runs/phase4_1_final_fix_smoke_20260721_a exists=False
runs/phase4_1_final_fix_smoke_20260721_b exists=False
```

Both standalone commands used required `--run-dir`, `configs/train.yaml`, smoke mode, and
seed 42. A exited 0 in `36.7s`; B exited 0 in `36.8s`. Each requested 5,000 steps,
completed 6,144 at the PPO rollout boundary, and performed five deferred 200-step
validation episodes with mean reward `-104.51`.

An independent strict parser verified config/metadata contracts, schema-2 file-identity
headers, path `stat` identity, exact role/worker inventory, complete JSONL, record count,
and SHA-256 from each artifact's single byte read. Both runs contain 31 train and 6
validation records. Every A/B train and validation tuple is equal, train values stay in
`[0, 10000)`, and validation values stay in `[10000, 11000)`. The first three train tuples
are `(42, 948, 2314)`, `(191664963, 3546, 2463)`, and
`(1662057957, 5299, 6974)`; the six validation tuples are `(42, 10746, 10418)`,
`(191664963, 10103, 10696)`, `(1662057957, 10595, 10383)`,
`(1405681631, 10643, 10445)`, `(942484272, 10361, 10833)`, and
`(929893137, 10640, 10594)`. Each TensorBoard event file contains 62 finite scalar values.

Final checkpoint hashes are
`b4850408ad0447cb682bece00a77767e8e5258843e223280624cef85df87ba70` (A) and
`0fbcf53ffa5b429480d267aba0d01207f003ed3b628bf1d7e7e40393fe73755a` (B).
Each reloaded at `num_timesteps=6144` and completed 100 deterministic real MetaDrive steps
with 101 finite observations, 100 finite rewards, no termination, and no truncation. The
post-audit `python`/`pythonw` process count was 0. Nothing was pushed.

## Phase 4.1 parent-held worker identity follow-up

This section supersedes every earlier Phase 4.1 gate and smoke result in this log. All old
run directories remain preserved. The final reviewer demonstrated that a post-close
replacement could write a valid schema-2 header containing its own new file identity, so
header/path agreement was still self-attestation rather than worker provenance.

Strict RED→GREEN remediation now exposes an immutable JSON/pickle-safe descriptor from
each still-open writer. Train and validation descriptors carry role, worker, workspace-
relative path, device, and inode through `DummyVecEnv.get_attr` or
`SubprocVecEnv.get_attr`. The parent validates exact expected workers and retains those
descriptors before either VecEnv closes. Inventory then requires each opened `fstat`, path,
strict header, role, and worker to match parent memory before one stable read supplies
parse/count/SHA-256. Self-recomputed replacement headers, replacement before or after
close, missing/extra artifacts, duplicate descriptors, and mismatched workers fail closed.
Descriptor collection failure prevents inventory and publication.

Subprocess close now performs bounded join and inspects every worker `exitcode`. Nonzero,
malformed, or unconfirmed exit status, any terminate/kill escalation, and any surviving
worker are cleanup failures. A worker-side `env.close()` failure that exits 1 without a
pipe error is therefore detected. Without a primary exception the failure blocks
publication; with a primary exception it is attached as a note. Successful and failed
first closes detach resources, so later closes do not retry them. A real two-worker SB3
integration fetched both descriptors with `closed=False` and observed exit codes `[0, 0]`.

### Superseding verification

- focused adversarial identity/close set: `29 passed, 16 warnings in 10.35s`;
- canonical full pytest: `678 passed, 18 warnings in 33.58s`;
- Ruff: `All checks passed!`; changed-file format check: `5 files already formatted`;
- mypy: `Success: no issues found in 54 source files`;
- strict coverage: `678 passed, 18 warnings in 45.51s`, 3,636 statements, 266 missed,
  1,016 branches, 162 partial branches, exact `90.33%`;
- real MetaDrive trio: `15 passed, 15 warnings in 14.62s`.

The first post-change coverage run completed 666 tests but failed the required threshold
at exact `89.94%`. Requirement-specific malformed/duplicate/mismatched descriptor and
unconfirmed exit-code cases raised the final passing gate above threshold. No warning was
suppressed. Process audits were 0 before the gates, after the real trio, and after smoke
artifact reload.

### New parent-held identity PPO smokes

Read-only preflight proved both new destinations absent and both prior final-fix runs still
present:

```text
runs/phase4_1_worker_identity_final_smoke_20260721_a exists=False
runs/phase4_1_worker_identity_final_smoke_20260721_b exists=False
runs/phase4_1_worker_identity_smoke_20260721_a exists=True
runs/phase4_1_worker_identity_smoke_20260721_b exists=True
```

Both required-`--run-dir` CLI commands exited 0. A completed in `44.4s`; B completed in
`43.6s`. Each requested 5,000 steps, completed 6,144 at the PPO rollout boundary, and ran
five deferred 200-step validation episodes at mean reward `-104.51`.

An independent one-read parser verified exact inventory, strict JSON without duplicate
keys, schema-2 headers, metadata/header/path identity agreement, role/worker, complete
records, counts, and SHA-256. Parent-held provenance is established by the RED→GREEN
control-channel tests and real two-worker integration; the smoke metadata records the
validated resulting identities. A has train identity
`(2393062996, 77687093572382381)` and validation identity
`(2393062996, 99079191802415328)`; B has train identity
`(2393062996, 20547673300163201)` and validation identity
`(2393062996, 55732045388995816)`. Each run contains 31 train and 6 validation records,
and every A/B seed tuple matches. The first three train tuples are `(42, 948, 2314)`,
`(191664963, 3546, 2463)`, and `(1662057957, 5299, 6974)`; validation tuples remain
`(42, 10746, 10418)`, `(191664963, 10103, 10696)`,
`(1662057957, 10595, 10383)`, `(1405681631, 10643, 10445)`,
`(942484272, 10361, 10833)`, and `(929893137, 10640, 10594)`. Each TensorBoard event file
contains 62 finite scalar values.

Final checkpoint SHA-256 is
`429bd30bb2b715710c7f1b531330d1613b427580f7785ef36dd9f50cf4b7cfc0` for A and
`1e0e9673c4e81160f6bd756922f5bbd368dfef5caace8aa8b7607eb0ce4edeca` for B. Each
reloaded at `num_timesteps=6144` and completed 100 deterministic real MetaDrive steps with
101 finite observations, 100 finite rewards, no termination, and no truncation. The final
`python`/`pythonw` process count was 0. Nothing was pushed.

## Phase 4.2 external-review remediation

This section supersedes the earlier reward, validation-topology, and contract-version
evidence while preserving all historical run directories. The 24-dimensional Coordinator
Observation remains unchanged; explicit validity slots are still deferred.

The review remediation removes scenario identity and episode seeds from agent-visible
`SceneObservation` and keeps them as `SceneFrame` metadata. Privileged actors now retain
truthful visible/occluded flags. Action-validity reward terms use the previous frame and
analysis, near-miss uses only next Nominal/Hazard physical-risk TTC, and the research
contract is version 3. `ScenarioRuntime` threads hook-returned immutable state through
`ScenarioTransition`. Shield monitor requirements and low-level control fail-safe status
are explicit in both `info` and `DecisionTrace`.

Single-environment training now uses a parent `DummyVecEnv` plus a one-worker validation
`SubprocVecEnv`, so periodic validation occurs during learning without two MetaDrive
engines in one process. Each scheduled evaluation reseeds validation with the same root
seed. Reward TensorBoard fields use SB3 `record_mean`, averaging across vector environments
and the logger dump interval instead of retaining the final sample.

### Verification

- canonical coverage gate: `708 passed, 25 warnings in 71.63s`;
- total coverage: `90.13%` across 3,709 statements and 1,064 branches;
- Ruff lint and format: all checks passed, 90 files formatted;
- mypy: no issues in 54 source files;
- focused RED→GREEN tests proved reward mean logging and public
  `ScenarioTransition` export.

The pre-push independent review found no Critical issues and two Important issues. The
validation documentation now distinguishes fixed AppConfig root `seed` from the
five-run policy/RNG `training.seed`, and `ScenarioState.parameters` now recursively copies
and freezes finite JSON-like mappings/sequences. Nested aliasing and malformed parameter
boundaries are covered by five additional tests before the final 708-test gate.

No warning was suppressed. The warnings are upstream Matplotlib deprecations plus SB3
advisories for the intentional complementary VecEnv topology and unmodified evaluation
wrapping.

### Contract-v3 PPO smokes

The first attempted pair
`phase4_2_review_fix_smoke_20260721_a/b` was excluded after metadata revealed that the
non-editable virtual-environment wheel was still contract v2. The environment was
resynchronized from the current source and verified to expose contract 3 and an
agent-visible `SceneObservation` without `scenario_id` or `seeds` before new destinations
were launched.

`phase4_2_review_fix_v3_smoke_20260721_c` completed in 42.0s and
`phase4_2_review_fix_v3_smoke_20260721_d` completed in 48.2s. Each requested 5,000
timesteps, evaluated at exactly step 5,000, and then completed 6,144 at the PPO rollout
boundary. Each five-episode validation reported mean reward `-64.71` and episode length
`200`. Both metadata files record `research_contract_version=3` and
`observation_schema_version=1`.

Each run contains 31 train and 6 validation seed records. Ignoring the descriptor-specific
identity header, the C/D validation JSONL records are byte-equal:
`(42, 10746, 10418)`, `(191664963, 10103, 10696)`,
`(1662057957, 10595, 10383)`, `(1405681631, 10643, 10445)`,
`(942484272, 10361, 10833)`, and `(929893137, 10640, 10594)`.

TensorBoard contains all ten reward-component means at steps 2,048, 4,096, 5,000, and
6,144. C and D values are identical. Mean jerk penalty is `-1.7162`, `-1.4787`,
`-1.3225`, and `-1.2281` respectively; this demonstrates that the logger no longer reports
an arbitrary final transition as the rollout statistic. The jerk formula and configured
scale were not changed without scenario-specific evidence.

Final checkpoint SHA-256 values are
`edc3323fd49ed227db3a99824705679d9f01fff05783d0fee27642c16782ab12` for C and
`1df8aa745fb2dd9e4bdf847568999671e813adf0f2c468db86c077b8f3f781f5` for D. Both
reload at `num_timesteps=6144` and complete 100 deterministic real MetaDrive decisions
with finite 24-dimensional observations and finite rewards. Each reward sum is
`-32.434590492396445`.

## Phase 4.2 comparison-validity remediation (contract v4)

This section supersedes the contract-v3 learning objective while leaving the
24-dimensional Observation schema unchanged. `RewardContext` no longer accepts Agent
analysis. Near-miss, hard-rule, and unnecessary-brake terms now consume only the fixed
privileged oracle stored in the pre/post `SceneFrame`; hidden actors participate in the
oracle TTC. Vehicle, object, sidewalk, and building collisions share the vehicle collision
penalty. The Shield now receives separately derived `expected_agent_ids` and
`failed_agent_ids`, so intentional B1/ablation omissions do not activate an Agent-failure
floor. Active low-level control fail-safe status closes the Gymnasium environment and raises
instead of entering PPO as a normal high-level Action transition. Scenario outcomes reject
non-booleans and simultaneous success/failure.

The first attempted pair, `phase4_2_oracle_v4_smoke_20260721_e/f`, is excluded: the
non-editable environment retained the already-installed contract-v3 wheel. After
`uv sync --reinstall-package mad-driving`, the installed package reported contract 4 and a
`RewardContext` with only frame/action/Shield/timing fields.

The authoritative runs are `phase4_2_oracle_v4_smoke_20260721_g/h`. Each requested 5,000
timesteps, evaluated five fixed validation episodes at step 5,000 with mean reward `-64.71`
and length `200`, and completed at the PPO rollout boundary of 6,144. Both metadata files
record `research_contract_version=4`, `observation_schema_version=1`, and shape `[24]`.
Ignoring descriptor identity headers, both runs have byte-equal sequences of 31 training
and 6 validation episode-seed records. Final checkpoint SHA-256 values are
`6c2dc8e226cb6b71bafeea51bc06d64f3f81ecd0069463d1f05f7619c2cfc25d` for G and
`37144849eda8d6a508e4906b2dac65b44e134911aed03d55ca370f6f0a47f7f3` for H. Both reload
at 6,144 timesteps and complete 100 deterministic real MetaDrive decisions with finite
observations/rewards and identical reward sum `-32.434590492396445`.

Final verification: `721 passed, 25 warnings in 67.21s`; total coverage `90.08%` over
3,779 statements and 1,110 branches; Ruff lint/format and mypy all pass. Warnings remain the
documented upstream Matplotlib deprecations and SB3 notices for the intentional complementary
training/validation VecEnv topology.
