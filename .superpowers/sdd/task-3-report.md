# Task 3 Report: Occluded Crossing Actor

## Delivered

Implemented the seeded Level-3 `occluded_crossing` scenario while preserving the
fixed 24-dimensional observation and four-action interfaces.

- Added strict configuration and the fixed Level-3 overlay.
- Added a real named MetaDrive 0.4.3 `Cyclist` (`crossing-cyclist`), a static
  off-lane `static-occluder`, and a visible same-lane `crossing-lead`.
- Sampled only from `scenario_parameter_seed`: conflict distance 20--40 m,
  crossing offset 6--12 m, speed 2--6 m/s, trigger 1--3 s, lead gap 35--55 m,
  lead speed fraction 0.80--1.00, and a 2 s survival window.
- Kept the cyclist in simulator truth from reset. The complete visible-ID
  allowlist includes current simulator actors and hides only the cyclist until
  its lateral reveal boundary.
- Computes conflict distance from current ego-lane coordinates on every context
  request; the observation context contains only static occlusion metadata,
  IDs, and distance, never hidden cyclist kinematics.
- Added `VelocityCommand` to release the stationary cyclist at its sampled
  trigger using the real MetaDrive velocity API.
- `crash_human` plus exact cyclist contact is a scenario failure. Any typed
  collision or off-road state suppresses success; unrelated collision signals
  are not labelled a cyclist failure. Success requires the cyclist to cross
  clear of the ego corridor and survive a further 2 seconds.
- Shared stable Actor-ID generation between the control adapter and snapshot
  builder so a complete logical allowlist agrees with privileged/visible frame
  construction.

## RED / GREEN evidence

1. Initial desired-runtime tests were written first.

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_occluded_crossing.py -q
   ```

   RED: exit 1 during collection. Expected failure:
   `ImportError: cannot import name 'OccludedCrossingScenarioConfig'`.

   GREEN after the minimal configuration, actor command, runtime, registry, and
   adapter implementation:

   ```text
   7 passed, 14 warnings in 4.44s
   ```

2. The Level-3 overlay selection test preceded the YAML file.

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/unit/config/test_loader.py -q -k occluded_crossing
   ```

   RED: exit 1 with expected `FileNotFoundError` for
   `configs/scenarios/occluded_crossing.yaml`.

   GREEN after adding the overlay:

   ```text
   1 passed, 20 deselected in 0.11s
   ```

3. The real headless smoke caught two pinned-MetaDrive integration defects:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration -k occluded
   ```

   RED #1: `scenario_road_geometry()` returned `None` because the return value
   had been accidentally placed after the new lane-position helper. GREEN after
   restoring the method boundary.

   RED #2: MetaDrive 0.4.3 `StaticDefaultVehicle` rejected construction without
   `vehicle_config`. GREEN after passing the required empty config.

   Final GREEN:

   ```text
   1 passed, 2 deselected, 14 warnings in 5.80s
   ```

4. Coverage was initially 89.98%, just below the configured gate. Added tests
   for current-lane conflict-distance recomputation, missing cyclist fail-fast,
   and unexpected spawn IDs. The focused regression command then reported:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_occluded_crossing.py -q
   ```

   ```text
   10 passed, 14 warnings in 4.15s
   ```

## Verification

| Command | Result |
| --- | --- |
| `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_occluded_crossing.py tests/unit/world_model/test_snapshot_builder.py tests/unit/agents/test_hazard.py -q` | 42 passed |
| `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_actor_manager.py tests/unit/scenarios/test_manager.py tests/unit/scenarios/test_factory.py tests/unit/config/test_loader.py -q` | 38 passed |
| `.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration -k occluded` | 1 passed, 2 deselected |
| `.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration` | 3 passed |
| `.venv\Scripts\python.exe -m ruff check src tests` | All checks passed |
| `.venv\Scripts\python.exe -m mypy src` | Success: no issues found in 63 source files |
| `.venv\Scripts\python.exe -m pytest -q --cov=mad_driving --cov-report=term-missing` | 792 passed, 25 third-party warnings, 90.05% coverage |
| `git diff --check` | exit 0 |

The full-suite warnings are existing Matplotlib deprecations and Stable-Baselines
evaluation-wrapper warnings; no test failures occurred.

## Real MetaDrive coverage

The `occluded` headless smoke verifies:

- real `Cyclist` construction, stable ID/name, and `crash_human` info-field
  support;
- cyclist hidden from `visible_actors` while present in privileged
  `all_actors` as `visible=False, occluded=True`;
- deterministic reveal after the lateral boundary;
- a finite privileged oracle TTC after ego movement;
- identical same-seed cyclist trajectories; and
- force-destroy cleanup across reset.

## Files changed

Created:

- `configs/scenarios/occluded_crossing.yaml`
- `src/mad_driving/scenarios/actor_ids.py`
- `src/mad_driving/scenarios/occluded_crossing.py`
- `tests/unit/scenarios/test_occluded_crossing.py`

Modified:

- `src/mad_driving/config/models.py`
- `src/mad_driving/envs/control_metadrive_env.py`
- `src/mad_driving/envs/multi_agent_speed_env.py`
- `src/mad_driving/scenarios/__init__.py`
- `src/mad_driving/scenarios/actor_manager.py`
- `src/mad_driving/scenarios/actors.py`
- `src/mad_driving/scenarios/manager.py`
- `src/mad_driving/world_model/snapshot_builder.py`
- `tests/integration/test_phase5_metadrive_headless.py`
- `tests/unit/config/test_loader.py`
- `tests/unit/scenarios/test_factory.py`

## Design decisions and concerns

- The Cyclist is the real MetaDrive 0.4.3 class at
  `metadrive.component.traffic_participants.cyclist.Cyclist`; the manager passes
  its stable name directly to the engine.
- The static occluder is positioned beyond the lane edge and displaced along the
  lane to keep it clear of the cyclist's physical crossing path. Occlusion is a
  deliberate logical observation boundary, not a camera/rendering model.
- Exact contact attribution uses the Task 1 manager's MetaDrive Bullet contact
  query, preventing unrelated vehicle/object collisions from becoming cyclist
  scenario failures.
- No Task 4 curriculum, provenance, contract, or documentation changes were
  introduced.

## Independent review remediation (2026-07-22)

### Fixed findings

- `cyclist_revealed` is now initialized and latched in immutable
  `ScenarioState.parameters` on first reveal-boundary entry. Once true, the
  logical allowlist always retains `crossing-cyclist`, including after far-side
  clearance and at the survival-success boundary.
- `cyclist_collision` is likewise latched in immutable state. Scenario failure
  now uses `crash_human OR exact cyclist contact OR prior latched collision`.
  Thus a transient typed signal or a transient Bullet contact cannot be lost.
  Unrelated vehicle/object collisions still only suppress success.
- The real smoke imports `Cyclist` and uses `isinstance`, verifies the visible
  same-lane lead and its sampled speed, continues through clearance/survival,
  and checks persistent cyclist visibility.
- A dedicated real collision regression uses MetaDrive 0.4.3's actual
  `Cyclist.set_position(ego.position)` overlap followed by one simulator step.
  It asserts `crash_human`, typed `scenario_failure`, and privileged
  `collision_kind == "crossing_actor"`.

### RED / GREEN evidence

1. Added unit regressions before changing runtime code:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_occluded_crossing.py -q
   ```

   RED: 3 failed, 9 passed. The failures were the expected missing
   `cyclist_revealed` state key, flag-only `crash_human` returning
   `failure=False`, and contact-only collision returning `failure=False`.

   GREEN after state latching and OR attribution:

   ```text
   12 passed, 14 warnings in 5.28s
   ```

2. Strengthened real integration coverage before exposing the sampled lead
   speed in state:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration -k occluded
   ```

   RED: expected `KeyError: 'lead_speed_mps'` from the real smoke. GREEN after
   preserving the actual sampled lead speed in `ScenarioState.parameters`:

   ```text
   2 passed, 2 deselected, 14 warnings in 7.66s
   ```

### Remediation verification

| Command | Result |
| --- | --- |
| `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_occluded_crossing.py tests/unit/world_model/test_snapshot_builder.py tests/unit/agents/test_hazard.py -q` | 47 passed |
| `.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration` | 4 passed |
| `.venv\Scripts\python.exe -m ruff check src tests` | All checks passed |
| `.venv\Scripts\python.exe -m mypy src` | Success: no issues found in 63 source files |
| `.venv\Scripts\python.exe -m pytest -q --cov=mad_driving --cov-report=term-missing` | 795 passed, 25 warnings, 90.05% coverage |
