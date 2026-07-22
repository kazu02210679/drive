# Phase 5 Final Whole-Branch Review Fix Report

Date: 2026-07-22

Starting HEAD: `18b02283a78b8528c331426d66a781d6177f534a`

Branch: `feat/phase5-scenarios`

## Outcome

All four final whole-branch review findings were fixed with tests written and observed
failing before each production change. The implementation preserves the accepted Phase 5
scenario integrity work, the 24-dimensional observation, all four actions, and Phase 4
behavior. No physics step or unsupported simulator lifecycle hook was added.

## Pinned MetaDrive 0.4.3 source inspected

The local pinned installation under `.venv/Lib/site-packages/metadrive` was used as the API
authority. The inspected implementation included:

- `component/vehicle/base_vehicle.py`: `before_step`, `after_step`, collision flags, and
  `set_velocity`/`last_velocity` behavior;
- `manager/traffic_manager.py` and `manager/base_manager.py`: manager ordering, object
  ownership, spawn, and cleanup;
- vehicle navigation/localization code used by `BaseVehicle`;
- `component/static_object/base_static_object.py` and
  `component/static_object/traffic_object.py`: `TrafficObject`, `TrafficBarrier`, static
  rigid-body construction, dimensions, pose, and collision group;
- collision callback/type dispatch showing traffic objects set `crash_object`, while a
  `StaticDefaultVehicle` sets `crash_vehicle`.

## TDD evidence and design decisions

### 1. Actor lifecycle, acceleration, and lane classification

RED tests were added first:

```text
.venv\Scripts\python.exe -m pytest \
  tests/integration/test_phase5_metadrive_headless.py::test_real_lead_brake_frame_reports_braking_acceleration_to_nominal_agent \
  tests/integration/test_phase5_metadrive_headless.py::test_real_cut_in_frame_relocalizes_merged_actor_to_the_ego_lane -q
```

Observed failures:

- Lead Brake exposed zero/non-braking longitudinal acceleration after the trigger because
  direct `set_velocity` reset MetaDrive's `last_velocity` reference;
- the physically merged Cut-in actor remained `same_lane=False` because navigation still
  referred to its spawn lane.

Implementation:

- `ScenarioActorManager` now captures each actor's pre-command velocity and publishes a
  manager-owned post-physics state after MetaDrive advances;
- `SceneSnapshotBuilder` consumes that state for scenario actors, including acceleration;
- the manager records the commanded lane reference, while `same_lane` still requires the
  actor's physical position to lie inside that lane, preventing premature Cut-in
  classification;
- the design deliberately avoids calling `BaseVehicle.before_step` for scripted actors:
  doing so would inject vehicle control and alter accepted deterministic trajectories.

GREEN evidence:

- the two focused real regressions passed;
- `tests/integration/test_phase5_metadrive_headless.py`: `12 passed`;
- scenario/world-model/config unit group: `230 passed`;
- Lead Brake reports finite sampled negative acceleration and changes the Nominal claim;
- merged Cut-in is seen by both Hazard and Nominal lead logic;
- deterministic trajectory and reset cleanup regressions remain green.

### 2. Absolute evaluation/checkpoint continuation

RED tests were added/updated first in `tests/integration/test_ppo_checkpoint.py`.

Observed failures:

- periodic sidecars held the pre-evaluation curriculum state when evaluation and checkpoint
  were due at the same model timestep;
- with two environments and non-aligned intervals, the old `n_calls`-relative scheduling
  produced evaluation events at 6/12/18/24 and checkpoints at 10/20 instead of crossing the
  absolute 7/11 deadlines; the required timestep-12 periodic checkpoint did not exist.

Implementation:

- evaluation and checkpoint callbacks use a repeating schedule anchored at absolute model
  timestep zero and advance to the first deadline strictly after the current restored
  timestep;
- VecEnv callback timesteps consume a deadline when the model timestep first reaches or
  crosses it, so no lossy integer frequency scaling is used;
- callback order is reward diagnostics, evaluation/curriculum advancement, then periodic
  checkpoint, making the sidecar a post-evaluation continuation point;
- smoke mode retains its requested end-of-smoke evaluation without changing normal absolute
  intervals.

The real PPO continuation test uses `num_envs=2`, evaluation interval 7, checkpoint interval
11, and compares uninterrupted training with a resume from the periodic timestep-12
checkpoint. Both paths produce evaluations at 8/14/22, checkpoints at 12/22, and identical
curriculum evaluations, levels, and final state. The resumed suffix is 14/22 and checkpoint
22, proving continuation through actual callback execution rather than sidecar parsing.

GREEN evidence:

- focused exact-sidecar and non-aligned continuation tests: `2 passed`;
- full `tests/integration/test_ppo_checkpoint.py`: `5 passed`;
- callback/training unit suites: `167 passed`.

### 3. Strict physical configuration validation

RED command:

```text
.venv\Scripts\python.exe -m pytest tests/unit/config/test_loader.py \
  -k "scenario_physical_ranges or crossing_start_minimum" -q
```

Observed result after ensuring every test supplied a complete range: `32 failed, 1 passed`.
Zero and negative minima were accepted for all scenario physical ranges, and crossing start
minima equal to or inside the reveal boundary were accepted.

Implementation:

- all sampled physical gaps, distances, speed fractions, speeds, triggers, merge duration,
  and deceleration magnitudes use a strictly-positive finite range model;
- all three survival durations remain strictly positive;
- the crossing start minimum must be strictly greater than `reveal_lateral_m`, preserving
  the hidden-at-reset invariant;
- documented maxima and dedicated scenario overlays are unchanged, including the Cut-in
  speed fraction maximum of 1.05 and the occluder lane-edge minimum of 0.5 m.

GREEN evidence: `tests/unit/config/test_loader.py`: `69 passed`. Validation occurs in
`load_config`/Pydantic before any MetaDrive environment is constructed.

### 4. Static occluder collision attribution

RED command:

```text
.venv\Scripts\python.exe -m pytest \
  tests/integration/test_phase5_metadrive_headless.py::test_real_occluder_contact_is_attributed_as_a_static_object_collision -q
```

Observed result: `1 failed`; the real collision ended as `crash vehicle` and
`info["crash_object"]` was `False`. The former vehicle object also exposed its built-in
4.515 m length instead of the configured 5.0 m.

Implementation:

- replaced `StaticDefaultVehicle` with `ScenarioOccluder`, a supported `TrafficBarrier`
  derivative in MetaDrive's `TrafficObject` collision group;
- retained stable ID, static pose, visible barrier model, exact 5.0 by 2.0 physical
  dimensions, logical occlusion, off-corridor placement, actor-state refresh, and cleanup.

GREEN result: the real contact regression passed with `crash_object=True`,
`crash_vehicle=False`, privileged `collision_kind="object"`, and actor type `obstacle`.

## Final verification

```text
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing
975 passed, 34 warnings in 106.05s
Total coverage: 90.11% (required: 90.0%)

.venv\Scripts\python.exe -m ruff check .
All checks passed!

.venv\Scripts\python.exe -m mypy --strict src
Success: no issues found in 65 source files

.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/base.yaml --smoke
exit 0; evaluation at 5000; timesteps=6144; best and final checkpoints written

git diff --check
exit 0
```

The generated CLI smoke run
`runs/phase5-smoke-20260722T070442.669928Z-3d0fd7620ce6a9443cd5241db0bac9d7`
was removed after its output was verified. Pre-existing run artifacts were not touched.

## Files changed

- `src/mad_driving/config/models.py`
- `src/mad_driving/scenarios/actor_manager.py`
- `src/mad_driving/scenarios/actors.py`
- `src/mad_driving/training/callbacks.py`
- `src/mad_driving/training/train.py`
- `src/mad_driving/world_model/snapshot_builder.py`
- `tests/integration/test_phase5_metadrive_headless.py`
- `tests/integration/test_ppo_checkpoint.py`
- `tests/unit/config/test_loader.py`
- `tests/unit/scenarios/test_actor_manager.py`
- `tests/unit/training/test_train.py`
- `.superpowers/sdd/final-review-fix-report.md`

## Concerns

There are no known functional blockers. The 34 warnings are existing third-party
PyParsing deprecations and Stable-Baselines3 warnings about Monitor/type differences between
training and isolated validation VecEnvs. Coverage passes at 90.11%, only 0.11 percentage
points above the configured threshold, so future untested source growth could reduce the
margin.
