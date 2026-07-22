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
- Stable level mapping is level 0 nominal, level 1 Lead Brake, level 2 uniform seeded
  Lead Brake/Cut-in selection, and level 3 Occluded Crossing with its existing secondary
  lead vehicle.
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

## Files changed

Production and documentation:

- `README.md`
- `src/mad_driving/cli/control_smoke.py`
- `src/mad_driving/cli/train.py`
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
- `tests/unit/envs/test_multi_agent_speed_env.py`
- `tests/unit/interfaces/test_models.py`
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
