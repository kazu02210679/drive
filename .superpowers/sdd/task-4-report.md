# Phase 5 Task 4 Report: Curriculum, provenance, resume, and verification

## Status and scope

Implemented Phase 5 Task 4 from starting HEAD `7e9ca74925162755783d24784bba26a91a57f192`
on `feat/phase5-scenarios`. The change preserves the existing Lead Brake, Cut-in,
Occluded Crossing, 24D observation, four-action policy, fixed validation reseeding, and
checkpoint transaction behavior. No Phase 6 baselines, ablations, reports, plots, or GIF
work was added.

## Lifecycle and schema decisions

- `CurriculumState(level, consecutive_passes, evaluations)` is frozen and rejects booleans,
  non-integers, negatives, level values outside `0..3`, and pass counts greater than the
  evaluation count.
- `CurriculumController` accepts only complete scheduled validation aggregates. Training
  observations are rejected and test observations are explicitly rejected. Automatic mode
  requires both success and collision thresholds for the configured consecutive count,
  advances exactly one level, clears the streak on advancement, caps at level 3, and never
  regresses. Fixed mode retains its configured level.
- Stable level mapping is level 0 nominal, level 1 Lead Brake, level 2 Lead Brake/Cut-in,
  and level 3 Occluded Crossing with its existing secondary lead vehicle. Fixed level 2
  accepts either concrete scenario or `auto` uniform seeded selection; automatic mode
  requires `auto`. The dedicated Cut-in overlay remains concrete.
- `CurriculumEvalCallback` extends the fixed-seed SB3 evaluation callback. It captures one
  strict Boolean `scenario_success`/`collision_occurred` terminal record per validation
  episode, observes only after a scheduled evaluation, atomically persists state, and calls
  `env_method("set_difficulty_level", level)` on training and validation vectors only when
  the level changes.
- Environments stage a pending difficulty level. Active scenario state is unchanged until
  the next reset.
- `curriculum_state.yaml` is written through a same-directory temporary file, YAML dump,
  flush, `fsync`, and `os.replace`. Resume validates the parent file and SHA-256, parses the
  exact state keys and strict values, checks level/config compatibility, restores the exact
  state before PPO loading, and broadcasts the restored level to both vectors.
- `EPISODE_SEED_ARTIFACT_SCHEMA_VERSION` is 3. Every reset record has exactly `role`,
  `worker_index`, `environment_seed`, `scenario_selection_seed`,
  `scenario_parameter_seed`, `scenario_id`, `difficulty_level`, and
  `scenario_parameters`. Parameters are recursively checked for string keys, JSON-safe
  values, and finite numbers. Existing JSONL header/parser, append/flush/`fsync`, record
  count, SHA-256, and file-identity architecture is preserved.
- Seed meanings are explicit: `environment_seed` identifies the Gymnasium episode RNG,
  `scenario_selection_seed` is the role-bounded MetaDrive scenario index, and
  `scenario_parameter_seed` drives deterministic Phase 5 choice/parameter sampling.
- `RESEARCH_CONTRACT_VERSION` is 5. `run_metadata.json` binds the curriculum path, state
  values, and SHA-256 alongside schema-v3 episode-seed descriptors.
- `DecisionTrace` accepts scenario identity and difficulty only as a complete pair; both are
  required whenever episode seed metadata is present. Scenario identity, parameters, and
  seed metadata remain outside the 24D agent observation.
- The control-smoke trace supplies its nominal level-0 identity, preserving its complete
  seed metadata under the v5 trace contract.
- Real MetaDrive worker shutdown retains strict close, exit-code, and escalation auditing,
  while allowing a five-second bounded graceful join before terminate/kill. This closes a
  full-suite load race without weakening failed-worker detection.

## RED/GREEN evidence

### Slice A: curriculum state machine

RED:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/training/test_curriculum.py -q
```

Collection failed with `ModuleNotFoundError: mad_driving.training.curriculum`.

GREEN after the initial implementation: `19 passed`. Final focused result after strict
artifact/config coverage was added: `22 passed, 14 warnings`.

### Slice B: callback and reset-boundary activation

RED:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/training/test_callbacks.py tests/unit/envs/test_multi_agent_speed_env.py -q
```

The first run failed because `CurriculumEvalCallback` did not exist. The next RED cases
failed because terminal environment info did not yet expose `collision_occurred`.

GREEN: `91 passed, 17 warnings`.

### Slices C and D: provenance, trace, atomic state, and exact resume

RED:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/training/test_episode_seeds.py tests/unit/training/test_metadata.py tests/unit/training/test_train.py tests/unit/interfaces/test_models.py -q
```

The first run failed on the missing curriculum artifact/metadata support and the old schema
and contract versions.

GREEN: `281 passed, 14 warnings`.

CLI overlay RED/GREEN:

```text
tests/unit/cli/test_train.py: 2 failed because --overlay was unrecognized
tests/unit/cli/test_train.py: 10 passed, 14 warnings after implementation
```

### Slice E: real integration and lifecycle corrections

The fixed-level replay test runs each level `0..3` twice and compares canonical JSON bytes
covering scenario identity, parameters, actor trajectories, outcomes, and trace metadata.
Its first complete run passed all eight Phase 5 integration cases.

The real fixed-Level-1 8-step PPO smoke passed (`1 passed, 16 warnings`). The real automatic
8-step PPO smoke with one scheduled validation passed and persisted level 1
(`1 passed, 16 warnings`).

The first combined integration run exposed two legacy tiny resume fixtures without the new
environment difficulty method. Their focused RED failed with `AttributeError`; the fixtures
were changed to stage difficulty until reset, and the focused resume pair passed
(`2 passed, 20 warnings`).

The first full suite exposed eight control-smoke traces with complete seed metadata but no
scenario pair. Adding the nominal level-0 pair made the focused control gate pass
(`16 passed, 14 warnings`).

Under the first coverage run that reached the automatic real smoke, a validation worker
occasionally exceeded the one-second graceful close window and was terminated with exit
code `-15`. A focused lifecycle regression reproduced the failure. After extending only
the bounded graceful join window, the cleanup regression group passed (`4 passed`) and the
real automatic smoke passed again.

## Final verification

```text
.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py tests/integration/test_rl_metadrive_headless.py tests/integration/test_ppo_checkpoint.py -q -m integration
20 passed, 26 warnings

.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
861 passed, 29 warnings in 105.08s
Total coverage: 90.12% (branch coverage enabled; required minimum 90%)

.venv\Scripts\python.exe -m ruff check src tests
All checks passed

.venv\Scripts\python.exe -m mypy --strict src
Success: no issues found in 64 source files

.venv\Scripts\python.exe -m mad_driving.cli.train --help
Exit 0; help includes repeatable --overlay, --smoke, --run-dir, and --resume-from

git diff --check
Exit 0
```

The exact final integration, deterministic replay, static-analysis, diff, and clean-tree
checks are rerun immediately before commit; the final commit hash is reported to the parent.

## Independent-review remediation

The remediation started from Task 4 commit `b9ac8e11a282dbc59853377703515693a46befe0`.
No prior Phase 5 commit was squashed and no push was performed.

### Checkpoint-bound exact curriculum resume

RED added a real 24-step automatic PPO run with evaluations/checkpoints at steps 8, 16,
and 24. The first run failed because
`ppo_checkpoint_8_steps.zip.curriculum.yaml` did not exist.

GREEN introduces an adjacent atomic sidecar for every periodic, best, and final checkpoint.
Periodic callbacks bind the pre-evaluation state at their own save point, best checkpoints
bind the post-evaluation state that selected them, and final checkpoints bind the final
post-learning state. `run_metadata.json` inventories each checkpoint/sidecar path, both
SHA-256 values, and exact state values. Resume selects the one descriptor matching the
requested checkpoint and never reads the run-final state as a substitute. The real test
restores `(0,0,0)`, `(1,0,1)`, `(2,0,2)`, best `(1,0,1)`, and final `(3,0,3)` from their
five exposed checkpoints.

```text
.venv\Scripts\python.exe -m pytest tests/integration/test_ppo_checkpoint.py::test_every_published_checkpoint_restores_its_exact_automatic_curriculum_state -q -m integration
1 passed, 15 warnings
```

### Immutable reads, duplicate keys, and state/config invariants

RED produced 14 focused failures for duplicate YAML/JSON keys, replacement-race handling,
state/config invariants, and selection compatibility. A separate correction RED produced
three failures proving fixed level 2 must retain concrete Lead Brake/Cut-in support.

GREEN uses one open descriptor and one byte snapshot for sidecar digest and parse. It checks
`fstat` stability, path/file identity, size/timestamps, and expected digest before close;
the race regression substitutes a different path identity between read and final identity
validation. YAML parsing rejects duplicate keys recursively and run metadata uses a strict
JSON object-pairs hook. Resume validates fixed streak zero; automatic below-cap streak,
minimum evaluations for advancement/current streak, and valid capped level-3 states.
Configuration rejects incompatible selection/curriculum combinations before environment
construction. Fixed level 2 accepts `lead_brake`, `cut_in`, or `auto`; automatic accepts
only `auto`.

```text
.venv\Scripts\python.exe -m pytest tests/unit/training/test_curriculum.py tests/unit/config/test_loader.py tests/unit/training/test_metadata.py -q
109 passed, 14 warnings
```

### Callback schedules, atomic failures, CLI, and shutdown

- Two real scheduled evaluations aggregate mixed episode outcomes independently. The test
  proves exactly one `observe` per schedule, stale-record clearing, exact record counts,
  one level-change broadcast, and missing/extra terminal-record rejection.
- Atomic-write failure tests inject `fsync`, `os.replace`, and cleanup `unlink` failures.
  The old destination is preserved, no partial destination is published, cleanup is
  attempted, and a secondary cleanup error is retained without masking the primary error.
- CLI RED failed on required `--run-dir` and missing allocation helpers. GREEN atomically
  reserves a collision-free directory under `training.run_root` when the option is omitted;
  explicit `--run-dir` behavior is unchanged.
- Multiworker shutdown RED showed the second worker still receiving a separate five-second
  join. GREEN uses one monotonic deadline across all graceful/terminate/kill joins; the
  escalation test proves the second worker receives zero remaining grace after the first
  consumes the shared budget.

```text
.venv\Scripts\python.exe -m pytest tests/unit/training/test_callbacks.py -q
16 passed, 20 warnings

.venv\Scripts\python.exe -m pytest tests/unit/training/test_curriculum.py -q
32 passed, 14 warnings

.venv\Scripts\python.exe -m pytest tests/unit/cli/test_train.py -q
11 passed, 14 warnings

.venv\Scripts\python.exe -m pytest tests/unit/training/test_train.py -q
127 passed, 14 warnings
```

### Strengthened real replay and final remediation gates

Real fixed-level replay now runs bounded complete episodes to environment/scenario terminal
outcomes and serializes every observation, reward, actor state, visibility set, dynamic
scenario state, outcome, and DecisionTrace metadata. Five repeated pairs cover level 0,
level 1, both seeded level-2 branches (`43` Lead Brake and `42` Cut-in), and level 3.
Assertions prove lead deceleration, Cut-in merge, cyclist movement/reveal, the secondary
lead, and successful scenario outcomes. Every loaded floating PPO policy tensor is checked
with `isfinite` before prediction.

```text
.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py tests/integration/test_rl_metadrive_headless.py tests/integration/test_ppo_checkpoint.py -q -m integration
22 passed, 27 warnings in 63.70s

.venv\Scripts\python.exe -m pytest tests/unit -q
858 passed, 20 warnings in 20.94s

.venv\Scripts\python.exe -m mad_driving.cli.train --help
Exit 0; --run-dir is optional and documents training.run_root allocation

.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/base.yaml --smoke
Exit 0; fresh run allocated automatically; 6,144 timesteps; best/final published

.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
888 passed, 33 warnings in 104.40s; branch coverage 90.02%

.venv\Scripts\python.exe -m ruff check .
All checks passed

.venv\Scripts\python.exe -m mypy --strict src
Success: no issues found in 65 source files
```

The generated exact-command smoke run was inspected and removed after verification. Its
warning categories were unchanged: Matplotlib/PyParsing deprecations plus SB3 notices for
unmonitored evaluation environments and intentionally different train/eval VecEnv types.

## Files changed

Production and documentation:

- `README.md`
- `src/mad_driving/cli/control_smoke.py`
- `src/mad_driving/cli/train.py`
- `src/mad_driving/config/loader.py`
- `src/mad_driving/config/models.py`
- `src/mad_driving/config/parsing.py` (new in remediation)
- `src/mad_driving/envs/multi_agent_speed_env.py`
- `src/mad_driving/interfaces/decision_trace.py`
- `src/mad_driving/training/__init__.py`
- `src/mad_driving/training/callbacks.py`
- `src/mad_driving/training/curriculum.py` (new)
- `src/mad_driving/training/episode_seeds.py`
- `src/mad_driving/training/metadata.py`
- `src/mad_driving/training/train.py`
- `.superpowers/sdd/task-4-report.md`

Tests:

- `tests/integration/test_phase5_metadrive_headless.py`
- `tests/integration/test_ppo_checkpoint.py`
- `tests/integration/test_rl_metadrive_headless.py`
- `tests/unit/cli/test_train.py`
- `tests/unit/config/test_loader.py`
- `tests/unit/envs/test_multi_agent_speed_env.py`
- `tests/unit/interfaces/test_models.py`
- `tests/unit/scenarios/test_manager.py`
- `tests/unit/training/test_callbacks.py`
- `tests/unit/training/test_curriculum.py` (new)
- `tests/unit/training/test_episode_seeds.py`
- `tests/unit/training/test_metadata.py`
- `tests/unit/training/test_train.py`

## Concerns

- The 29 warnings are the same pre-existing third-party warning categories: Matplotlib
  pyparsing deprecations and SB3 warnings about different train/eval vector types or an
  evaluation environment without `Monitor`. New integration coverage increases the number
  of occurrences but introduces no new warning category.
- MetaDrive requires train/evaluation engine isolation in this process model. The automatic
  smoke therefore retains subprocess validation; strict worker close and exit auditing is
  covered explicitly.
- No test-role seed appears in training or validation artifacts, checkpoint selection, or
  curriculum observations.

## Second independent-review remediation

This remediation started from `487c5d560f48530ee196f98098d3150fac34baf1` and preserves
the per-checkpoint curriculum sidecar architecture from the first review fix.

### Immutable checkpoint and resolved-config snapshots

RED proved that a contract-valid checkpoint B could replace checkpoint A only while
`PPO.load` reopened the path, then A could be restored before the post-load hash check.
The run succeeded after loading B. Two more RED cases showed root and nested duplicate keys
in `config_resolved.yaml` were silently accepted.

```text
.venv\Scripts\python.exe -m pytest tests/unit/training/test_train.py -k "authenticated_checkpoint_snapshot or duplicate_resolved_config" -q
3 failed, 127 deselected
```

GREEN stores checkpoint bytes in the validated immutable `ResumeSource`, hashing and
identity-checking the same descriptor snapshot before passing a new `BytesIO` over those
exact bytes to SB3. The mutable source path is never reopened for model loading. Resolved
config now uses the same stable descriptor reader and recursive unique-key YAML parser.

```text
3 passed, 127 deselected
```

### Strict sidecar and episode-seed numeric types

Sidecar RED accepted Boolean and floating `schema_version` values. Episode-seed RED accepted
floating schema/worker/file-identity values and Boolean/floating worker identities through
Python numeric equality.

```text
tests/unit/training/test_curriculum.py -k schema_version_requires: 2 failed
tests/unit/training/test_episode_seeds.py -k non_integer_numeric: 5 failed, 6 passed
```

GREEN requires exact Python `int` values after YAML/JSON decoding. Header schema, worker,
device, and inode fields are validated before equality; record worker and all three seed
fields are likewise validated before canonical identity comparison.

```text
tests/unit/training/test_curriculum.py -k schema_version_requires: 2 passed
tests/unit/training/test_episode_seeds.py -k non_integer_numeric: 11 passed
```

### Shared shutdown budget and completed audit semantics

RED showed workers that require a positive post-terminate join were killed because the
graceful phase consumed the entire deadline. It also showed the close audit marker was set
while a worker remained alive, and later that an unconfirmed `None` exit code was marked
complete.

```text
tests/unit/training/test_train.py -k "multiworker_shutdown or failed_worker_shutdown":
2 failed, 1 passed
test_subprocess_none_exitcode_after_escalation_is_unconfirmed: 1 failed
```

GREEN divides one absolute five-second deadline into bounded 3/1/1-second
graceful/terminate/kill phases. Each phase assigns an equal positive share to its workers;
timeouts do not multiply with worker count. The audit marker is written only after every
worker is not alive and has a confirmed integral exit code.

```text
tests/unit/training/test_train.py -k "multiworker_shutdown or failed_worker_shutdown":
3 passed
focused close-audit confirmation group: 3 passed
```

### Atomic sidecars and safe implicit-directory cleanup

The existing durability injection tests are now parameterized across both
`write_curriculum_state` and `write_checkpoint_curriculum_state`: six cases cover `fsync`,
`os.replace`, and `unlink` cleanup failures, preserving the old destination, proving no
partial publish, and retaining primary plus cleanup errors (`6 passed`).

Implicit-directory RED left an unchanged empty reservation behind after a pre-ownership
failure. GREEN records its device/inode/creation identity and removes it only if the same
directory is still empty after a second identity check. Replacement, non-empty, and explicit
user paths are retained. A malformed resume through the real pre-ownership path is also
covered.

```text
tests/unit/cli/test_train.py focused RED: 2 failed, 2 passed
tests/unit/cli/test_train.py focused GREEN: 4 passed
malformed real resume cleanup: 1 passed
```

### Real PPO strengthening and documentation

Both real MetaDrive 8-step smokes now load best and final checkpoints. Every floating policy
tensor must be finite, and two deterministic predictions from the same finite 24D
observation must be byte-equivalent and within the four-action contract.

```text
.venv\Scripts\python.exe -m pytest tests/integration/test_rl_metadrive_headless.py::test_real_single_metadrive_training_uses_isolated_subprocess_validation tests/integration/test_rl_metadrive_headless.py::test_real_automatic_curriculum_advances_after_one_scheduled_validation -q -m integration
2 passed, 18 warnings
```

README training commands now show the exact no-`--run-dir` smoke, identify research contract
v5, and include `curriculum_state.yaml` plus periodic/best/final checkpoint sidecars in the
artifact tree. Historical Phase 4 v4 evidence is explicitly labeled historical.

### Second-remediation final verification

```text
.venv\Scripts\python.exe -m pytest tests/unit -q
884 passed, 20 warnings

.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py tests/integration/test_rl_metadrive_headless.py tests/integration/test_ppo_checkpoint.py -q -m integration
22 passed, 27 warnings in 56.88s

.venv\Scripts\python.exe -m mad_driving.cli.train --help
Exit 0; --run-dir remains optional and all expected options are listed

.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/base.yaml --smoke
Exit 0; 6,144 timesteps; fresh best/final checkpoints published under training.run_root

.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
913 passed, 33 warnings in 109.44s; total branch coverage 90.04%

.venv\Scripts\python.exe -m ruff check .
All checks passed

.venv\Scripts\python.exe -m mypy --strict src
Success: no issues found in 65 source files

git diff --check
Exit 0; only Git's configured LF-to-CRLF notices were emitted
```

The complete level 0--3 deterministic replay and both Level-2 seeded branches are part of
the 22-test integration command. The exact CLI smoke generated one isolated run; its path
was resolved and verified beneath the workspace `runs` root, then only that generated
artifact directory was removed. Warning categories remain unchanged: Matplotlib/PyParsing
deprecations and SB3 notices about unmonitored evaluation or intentionally different
train/evaluation VecEnv classes.

Second-remediation files changed:

- `.superpowers/sdd/task-4-report.md`
- `README.md`
- `src/mad_driving/cli/train.py`
- `src/mad_driving/training/curriculum.py`
- `src/mad_driving/training/episode_seeds.py`
- `src/mad_driving/training/metadata.py`
- `src/mad_driving/training/train.py`
- `tests/integration/test_rl_metadrive_headless.py`
- `tests/unit/cli/test_train.py`
- `tests/unit/training/test_curriculum.py`
- `tests/unit/training/test_episode_seeds.py`
- `tests/unit/training/test_train.py`
