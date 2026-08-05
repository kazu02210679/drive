# Task 2: Cut-in report

## Delivered

Implemented the seeded `cut_in` Phase 5 scenario.

- Added strict `CutInScenarioConfig` defaults: 20–40 m initial gap, 1–3 s trigger,
  1.5–3 s merge duration, 0.75–1.05 speed fraction, and a 3 s survival window.
- Added the level-2 `cut_in` overlay and registered the runtime in
  `ScenarioManagerRuntime`; the existing no-runtime fail-fast behavior remains unchanged.
- Added bounded cubic `smoothstep`, deterministic parameter/lane sampling, adjacent-lane
  fail-fast, lane-pose interpolation into the ego lane, continuous longitudinal targets,
  collision failure, and merge-plus-survival success.
- Added `LanePoseCommand` and the validated `ScenarioActorCommand` union without changing
  Lead Brake longitudinal command semantics.
- Extended simulator-neutral road geometry with sorted, queried adjacent lane indices and
  actual lane width. The MetaDrive binding queries its road network before reporting an
  adjacent lane.
- Force-destroys scenario-owned actors on reset so a previous Cut-in actor cannot be reused.
- Added unit and real MetaDrive deterministic-replay/reset-cleanup coverage. The 24D
  observation contracts remain untouched.

## Files

Created:

- `configs/scenarios/cut_in.yaml`
- `src/mad_driving/scenarios/cut_in.py`
- `tests/unit/scenarios/test_cut_in.py`

Modified:

- `src/mad_driving/config/models.py`
- `src/mad_driving/envs/control_metadrive_env.py`
- `src/mad_driving/envs/multi_agent_speed_env.py`
- `src/mad_driving/scenarios/__init__.py`
- `src/mad_driving/scenarios/actor_manager.py`
- `src/mad_driving/scenarios/actors.py`
- `src/mad_driving/scenarios/manager.py`
- `src/mad_driving/scenarios/parameters.py`
- `tests/integration/test_phase5_metadrive_headless.py`
- `tests/unit/config/test_loader.py`
- `tests/unit/scenarios/test_actor_manager.py`
- `tests/unit/scenarios/test_manager.py`

## RED/GREEN record

1. RED: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_cut_in.py -q`
   - Expected failure: `ModuleNotFoundError: mad_driving.scenarios.cut_in`.
   GREEN after adding `smoothstep`: 3 passed.
2. RED: same focused test after specifying the runtime/config behavior.
   - Expected failure: `ImportError: cannot import name 'CutInScenarioConfig'`.
   GREEN after the minimal runtime/config/geometry path: 4 passed.
3. RED: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_actor_manager.py -q`
   - Expected failure: `LanePoseCommand` rejected by `ScenarioActorManager`.
   GREEN after command-union dispatch: 11 passed across Cut-in and actor-manager tests.
4. RED: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_manager.py -q`
   - Expected failure: `no registered runtime for selected scenario: cut_in`.
   GREEN after runtime registration: 17 focused scenario tests passed.
5. RED: `.venv\Scripts\python.exe -m pytest tests/unit/config/test_loader.py -q`
   - Expected failure: missing `configs/scenarios/cut_in.yaml`.
   GREEN after adding the overlay: 37 focused loader/scenario tests passed.
6. RED: `.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration -k cut_in`
   - First exposed a collision from an unsafe keep-speed smoke driver; the smoke was changed
     to request `STOP` while observing the scripted merge.
   - Second exposed MetaDrive object reuse after reset.
   GREEN after force-destroy reset cleanup: 1 passed, 1 deselected.
7. RED: focused Cut-in test after adding a public API assertion.
   - Expected failure: `CutInRuntime` missing from `mad_driving.scenarios`.
   GREEN after export: 16 focused scenario tests passed.

## Verification

- `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_cut_in.py tests/unit/scenarios/test_manager.py -q` — passed.
- `.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration -k cut_in` — passed.
- `.venv\Scripts\python.exe -m ruff check src tests` — passed (`All checks passed!`).
- `.venv\Scripts\python.exe -m mypy src` — passed (`Success: no issues found in 61 source files`).
- `.venv\Scripts\python.exe -m pytest -q` — passed: **766 passed**. The suite emitted 25
  pre-existing third-party deprecation/evaluation-wrapper warnings; no test failures.
- `git diff --check` — passed.

## Self-review

Standards review used the repository's `pyproject.toml` Ruff/mypy/pytest configuration and
the requested task brief. No violations or new smell findings remain after static checks.

Spec review confirmed:

- Cut-in uses only actual, stably sorted adjacent lanes queried from MetaDrive.
- Lane pose is a separate frozen command and Lead Brake's acceleration path is preserved.
- No no-op fallback, Occluded Crossing, curriculum progression, contract v5, or Phase 6 work
  was added.
- Real MetaDrive replay verifies adjacent start, ego-lane arrival, finite state, identical
  first 20 positions for the same seed, and old-actor absence after reset.

## Concerns

- The MetaDrive lane query deliberately relies on the pinned MetaDrive 0.4.3 procedural
  `NodeRoadNetwork.graph` layout. The real headless integration test exercises this binding.
- The full suite reports existing third-party warnings, but all tests and strict static checks
  pass.

## Review follow-up: timing and collision attribution

### Diagnosis

The Cut-in lane-pose command is issued from `before_step(step_index=k)`, where physics has
already completed `k - 1` decision intervals. The old command used `k * dt`, producing a
one-interval forward jump on the first merge pose.

Pinned MetaDrive 0.4.3 provides exact contact-pair attribution. Its
`BaseVehicle._state_check()` uses `engine.physics_world.dynamic_world.contactTest(
vehicle.chassis.node(), True)`, and `metadrive.utils.utils.get_object_from_node()` resolves a
contact node back to its Python actor. The adapter now uses this exact dynamic-world pair query
and compares the resolved counterpart by identity with the requested scenario actor.

### RED/GREEN evidence

1. RED timing regression:

   ```text
   .venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_cut_in.py -q
   FAILED test_cut_in_lane_pose_starts_at_completed_time_without_a_longitudinal_jump
   Obtained: 40.0
   Expected: 39.0 ± 3.9e-05
   ```

   GREEN after anchoring lane poses to `(step_index - 1) * decision_interval_s`:

   ```text
   11 passed, 14 warnings in 4.04s
   ```

2. RED collision-attribution regressions:

   ```text
   .venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_cut_in.py tests/unit/scenarios/test_lead_brake.py tests/unit/scenarios/test_actor_manager.py -q
   4 failed, 26 passed, 14 warnings in 3.38s
   ```

   The failures showed that generic `raw_info["crash_vehicle"]` marked both scenarios as failed
   without an actor contact, and that `ScenarioActorManager.ego_collided_with` did not yet exist.

   GREEN after adding `scenario_ego_collided_with(actor_id)`, exact dynamic-world contact-pair
   lookup, and the two-signal failure predicate:

   ```text
   30 passed, 14 warnings in 5.80s
   ```

3. Final focused regressions, including a true contact with a false typed collision outcome:

   ```text
   .venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_cut_in.py tests/unit/scenarios/test_lead_brake.py tests/unit/scenarios/test_actor_manager.py -q
   32 passed, 14 warnings in 3.39s

   .venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration
   2 passed, 14 warnings in 6.15s

   .venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_cut_in.py tests/unit/scenarios/test_lead_brake.py tests/unit/scenarios/test_actor_manager.py tests/unit/envs/test_multi_agent_speed_env.py -q
   106 passed, 14 warnings in 3.78s
   ```

   The real Cut-in smoke calls `scenario_ego_collided_with("cut-in")` at reset and verifies no
   false-positive contact before the scripted merge.

### Final verification

```text
.venv\Scripts\python.exe -m ruff check src tests
All checks passed!

.venv\Scripts\python.exe -m mypy src
Success: no issues found in 61 source files

.venv\Scripts\python.exe -m pytest -q
773 passed, 25 warnings in 61.20s

git diff --check
exit 0
```

### Follow-up concerns

- The exact attribution intentionally relies on MetaDrive 0.4.3's documented-in-source Bullet
  `dynamic_world.contactTest`/`get_object_from_node` behavior; the pinned dependency and unit
  contact-pair tests make this explicit.
- Existing third-party matplotlib and Stable-Baselines evaluation-wrapper warnings remain, with
  no test or static-check failures.

## Review follow-up: physical-collision success boundary

### Change

Added `typed_collision_flags(raw_info)` in `scenarios.runtime`. It strictly validates and
returns the project-supported raw flags: `crash_vehicle`, `crash_human`, `crash_object`,
`crash_sidewalk`, and `crash_building`.

- Cut-in and Lead Brake retain actor-specific failure: only `crash_vehicle` plus exact contact
  with the owned scenario actor yields `scenario_failure`.
- Their success predicates now require no active typed physical-collision flags and no off-road
  outcome at the survival boundary.
- Nominal has no scenario actor, so physical collisions suppress success but never produce
  `scenario_failure`; outer-environment physical-collision termination remains authoritative.

### RED/GREEN evidence

1. Cut-in boundary RED:

   ```text
   .venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_cut_in.py -q
   FAILED test_cut_in_does_not_succeed_at_the_boundary_after_another_vehicle_collision
   Obtained: ScenarioStepResult(success=True, failure=False)
   Expected: success=False, failure=False
   1 failed, 13 passed, 14 warnings in 4.38s
   ```

   GREEN after the shared helper and Cut-in physical-collision success guard:

   ```text
   14 passed, 14 warnings in 3.75s
   ```

2. Lead Brake and Nominal boundary RED:

   ```text
   .venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_cut_in.py tests/unit/scenarios/test_lead_brake.py tests/unit/scenarios/test_runtime.py -q
   FAILED test_lead_brake_does_not_succeed_at_the_boundary_after_another_vehicle_collision
   FAILED test_nominal_does_not_succeed_or_fail_after_a_physical_collision
   2 failed, 46 passed, 14 warnings in 6.56s
   ```

   GREEN after applying `typed_collision_flags` consistently:

   ```text
   48 passed, 14 warnings in 7.90s
   ```

3. Focused environment and real Phase 5 verification:

   ```text
   .venv\Scripts\python.exe -m pytest tests/unit/envs/test_multi_agent_speed_env.py -q
   76 passed, 14 warnings in 3.15s

   .venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration
   2 passed, 14 warnings in 5.88s
   ```

### Final verification

```text
.venv\Scripts\python.exe -m ruff check src tests
All checks passed!

.venv\Scripts\python.exe -m mypy src
Success: no issues found in 61 source files

.venv\Scripts\python.exe -m pytest -q
780 passed, 25 warnings in 57.66s

git diff --check
exit 0
```
