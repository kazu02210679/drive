# Phase 5 Scenarios and Curriculum Design

**Date:** 2026-07-22  
**Status:** Approved direction; detailed design awaiting review  
**Parent:** Phase 4 branch `feat/phase4-rl-environment`, PR #4  
**Implementation branch:** `feat/phase5-scenarios`

## 1. Purpose

Phase 5 adds the three seeded MVP hazard scenarios and curriculum required by the
top-level specification:

1. Lead Brake;
2. Cut-in;
3. Occluded Crossing Actor;
4. Levels 0-3 curriculum progression.

The implementation plugs these features into the existing Phase 4
`ScenarioRuntime` lifecycle. It does not change the 24-dimensional Coordinator
observation schema.

## 2. Scope and non-goals

In scope:

- deterministic scenario parameter generation from `scenario_parameter_seed`;
- dedicated scripted Actors in the real MetaDrive 0.4.3 simulator;
- typed, strictly validated YAML configuration;
- scenario success and failure outcomes;
- Actor visibility and occlusion metadata;
- fixed-level and automatic curriculum modes;
- Actor-generation conditions in the durable per-worker JSONL artifact;
- unit, integration, reproducibility, and headless smoke tests.

Out of scope:

- Phase 6 baselines, ablations, reports, plots, GIFs, and final test-set runs;
- image recognition or camera observations;
- pedestrian Actors;
- lane-changing control for the ego vehicle;
- changes to the fixed 24-dimensional observation;
- using the test seed range for curriculum decisions.

## 3. Approaches considered

### A. `ScenarioRuntime` manipulates raw engine objects directly

This is the smallest code change, but objects spawned outside a MetaDrive manager
do not have a reliable reset/cleanup owner. Calling vehicle lifecycle methods from
the outer Gym wrapper would also risk ordering mistakes.

### B. Put all scenario logic in a custom MetaDrive manager

This follows MetaDrive's internal lifecycle, but would mix curriculum, random
parameter generation, outcome policy, and simulator details. Most logic would be
hard to test without Panda3D.

### C. `ScenarioRuntime` plus a thin MetaDrive Actor manager — selected

Pure Python runtimes own scenario meaning, parameter generation, visibility, and
outcomes. A small MetaDrive manager owns object spawning, commands, physics-step
ordering, and cleanup. The runtime accesses it through a narrow environment
adapter rather than importing MetaDrive classes.

This keeps seed and outcome logic simulator-independent while using MetaDrive's
supported manager lifecycle for real Actors.

## 4. Architecture

```text
CurriculumController
        |
        v
ScenarioRuntimeFactory -> ScenarioManagerRuntime -> selected scenario runtime
                                                   |
                                                   v
                                      ScenarioActorAdapter protocol
                                                   |
                                                   v
                                      MetaDrive ScenarioActorManager
                                                   |
                                                   v
                             scripted vehicles / cyclist / static occluder
```

### 4.1 Scenario manager runtime

`ScenarioManagerRuntime` implements the existing `ScenarioRuntime` protocol. On
each reset it:

1. reads the pending difficulty level;
2. creates a local NumPy generator only from `scenario_parameter_seed`;
3. chooses an allowed scenario using a stable, documented ordering;
4. samples all parameters once;
5. stores the selected scenario, level, and sampled parameters in immutable
   `ScenarioState`;
6. delegates lifecycle hooks to the selected runtime.

No global NumPy or Python random state is read. Scenario selection and parameter
sampling do not use `episode_rng_seed` or `metadrive_scenario_index`.

Stable scenario ordering is:

```text
level 0: nominal
level 1: lead_brake
level 2: lead_brake, cut_in
level 3: lead_brake, cut_in, occluded_crossing
```

Level 0 uses a nominal no-hazard runtime. At levels with multiple choices, the
choice is sampled uniformly before scenario-specific parameters are sampled.

### 4.2 MetaDrive Actor manager

`ScenarioActorManager` is registered by the concrete control MetaDrive
environment during engine setup. It owns every Phase 5 Actor and therefore clears
them during `before_reset()` and `destroy()`.

The manager exposes only these operations through the environment adapter:

- spawn a lane-relative vehicle;
- spawn a world-relative crossing cyclist;
- spawn a static occluder outside the ego lane;
- assign a validated scripted command or trajectory sample;
- query immutable Actor state needed by the runtime;
- enumerate scenario Actor IDs.

The runtime submits commands before `environment.step()`. The MetaDrive manager
applies them in its own `before_step()` and updates Actors in `after_step()`. A
missing Actor, duplicate ID, malformed command, or non-finite state is an internal
error and fails the episode immediately; it is not converted into a normal
scenario failure transition.

### 4.3 Coordinate contract

Lead Brake and Cut-in are generated relative to the ego vehicle's current lane
after simulator reset. They do not depend on hard-coded MetaDrive node names.

Occluded Crossing defines a conflict point in the ego-lane coordinate frame and
converts it once to world coordinates after reset. The crossing trajectory is
perpendicular to the local ego-lane heading. This permits a controlled crossing
on a procedural road without requiring the procedural road itself to contain an
intersection.

Every conversion is checked for finite values and lane availability. Missing or
unsupported geometry is an internal configuration/runtime error.

## 5. Scenario definitions

All ranges below are YAML defaults and are strictly validated. Lower bounds may
not exceed upper bounds.

### 5.1 Nominal level-0 runtime

- no dedicated hazard Actor;
- low background traffic density;
- success after the configured survival duration without collision, off-road, or
  internal error.

This runtime is distinct from the Phase 4 no-op runtime because it reports a
meaningful success outcome and curriculum metadata.

### 5.2 Lead Brake

Sampled defaults:

- initial longitudinal gap: 35-55 m;
- lead speed fraction of ego speed: 0.80-1.00;
- brake trigger: 1.0-3.0 s after reset;
- level-1 deceleration magnitude: 2.0-4.0 m/s²;
- level-2/3 deceleration magnitude: 4.0-8.0 m/s²;
- post-trigger survival window: 4.0 s.

The lead vehicle stays in the ego lane. Before the trigger it tracks its sampled
speed. At the trigger it decelerates at the sampled rate, bounded at zero speed.

Success means the brake event occurred and the survival window completed without
collision or off-road termination. A collision with the scenario Actor is
failure. Simulator/internal faults raise rather than becoming failure labels.

### 5.3 Cut-in

Sampled defaults:

- initial longitudinal lead: 20-40 m;
- source side: any available adjacent lane, selected deterministically;
- trigger: 1.0-3.0 s;
- merge duration: 1.5-3.0 s;
- speed fraction of ego speed: 0.75-1.05;
- post-merge survival window: 3.0 s.

The Actor follows a smooth cubic lateral trajectory with zero lateral velocity at
both ends. Longitudinal motion is continuous throughout the merge. If neither
adjacent lane exists, reset fails with an explicit geometry error instead of
silently changing the scenario.

Success means the merge completed and the survival window completed without a
collision. Collision with the cut-in Actor is failure.

### 5.4 Occluded Crossing Actor

MVP uses a cyclist Actor, not a pedestrian. This preserves the specified
`crossing_actor` semantic and MetaDrive's crossing-Actor collision signal.

Sampled defaults:

- conflict point: 20-40 m ahead of ego;
- crossing start offset: 6-12 m from the ego-lane centerline;
- crossing speed: 2.0-6.0 m/s;
- release trigger: 1.0-3.0 s;
- occluder offset from lane edge: at least 0.5 m;
- post-crossing survival window: 2.0 s.

A static occluder is placed outside the ego lane and outside the ego collision
corridor. The cyclist exists in privileged truth from reset, but its ID is
excluded from `visible_actor_ids` while it remains behind the configured
occlusion boundary. It becomes visible deterministically when it crosses that
boundary. The runtime provides `OcclusionRegion` and
`distance_to_conflict_point_m` through `ScenarioObservationContext`.

Success means the cyclist clears the ego corridor and the survival window
completes without collision. Collision with the cyclist is failure. The
occluder's collision behavior remains a normal static-object collision and uses
the Phase 4 static collision penalty.

## 6. Outcomes and termination

Scenario outcomes remain mutually exclusive. Scenario runtimes never infer
collision or off-road state from Agent Claims. They use typed simulator state or
the raw, validated MetaDrive collision fields.

The outer environment continues to terminate on collision, off-road, arrival,
scenario success, or scenario failure. Raw simulator termination without one of
those typed outcomes remains fatal.

## 7. Curriculum

Configuration supports two modes:

- `fixed`: always use one configured level;
- `automatic`: begin at a configured level and progress after validation.

Automatic progression uses validation episodes only. Test episodes are rejected
as curriculum input. After each scheduled validation:

1. compute scenario success rate and collision rate from typed episode info;
2. require success rate >= 0.80 and collision rate <= 0.05 by default;
3. require the condition for two consecutive scheduled validations by default;
4. advance by exactly one level, up to level 3;
5. reset the consecutive-pass counter;
6. broadcast the new pending level to training and validation VecEnvs;
7. apply it only on each environment's next reset.

There is no automatic regression in the MVP. The current level and advancement
decision are written to training metadata so resume restores the exact state.
Best-checkpoint comparison continues to use the fixed validation seed sequence.

## 8. Configuration

`AppConfig.scenarios` keeps the existing train/validation/test seed ranges and
adds strict Phase 5 settings:

```yaml
scenarios:
  train: {seed_start: 0, seed_count: 10000}
  validation: {seed_start: 10000, seed_count: 1000}
  test: {seed_start: 20000, seed_count: 1000}
  curriculum:
    mode: automatic
    fixed_level: 0
    initial_level: 0
    success_rate_threshold: 0.80
    collision_rate_threshold: 0.05
    consecutive_evaluations: 2
  lead_brake: {...}
  cut_in: {...}
  occluded_crossing: {...}
```

The three required `configs/scenarios/*.yaml` files are complete validated
overlays. `load_config()` is extended to accept an ordered base file plus zero or
more overlays, merge mappings recursively, reject type conflicts, and run final
`AppConfig` validation. The final merged configuration remains the only value
written as `config_resolved.yaml`.

## 9. Logging and contract version

Reset info and step info add:

- `scenario_id` (the selected concrete scenario);
- `difficulty_level`;
- `scenario_parameters` (deeply immutable internally, JSON-safe copy externally);
- `scenario_success` and `scenario_failure`.

The existing durable `episode_seeds/<role>-worker-NNN.jsonl` record is expanded to
include the selected scenario, level, and parameters returned by the actual reset.
Its schema version is incremented. The existing exclusive-create, descriptor,
file-identity, append/fsync, strict parse, count, and SHA-256 protections remain
unchanged.

`DecisionTrace` adds concrete scenario ID and difficulty level. Full parameters
remain in the per-episode JSONL record to avoid duplicating them on every step.
This is a schema change, so the experiment contract version is incremented. The
Coordinator observation schema remains version 1 and shape 24.

## 10. Failure handling

The following fail fast and close the environment:

- unsupported scenario ID or level;
- missing lane or adjacent lane required by the selected scenario;
- failed Actor spawn or duplicate Actor ID;
- Actor missing after spawn;
- non-finite Actor state or trajectory;
- malformed visibility set;
- curriculum update from test data;
- curriculum state that cannot be restored on resume;
- any Scenario Actor manager lifecycle or cleanup error.

Expected physical outcomes such as a collision remain normal episode outcomes.

## 11. Test strategy

Implementation follows strict red-green-refactor TDD.

Unit tests cover strict config validation, deterministic selection and sampling,
each runtime's trigger/trajectory/visibility/outcomes, adapter ordering and
cleanup, curriculum progression, test-role rejection, resume state, and JSONL
schema validation.

Integration tests cover real headless spawning and cleanup, privileged versus
Agent-visible occlusion, identical-seed trajectory replay, no Actor leakage across
resets, a Level 0-3 curriculum smoke, and a short PPO smoke using the Phase 5
runtime factory.

Final verification includes the complete pytest suite with >=90% coverage, Ruff,
strict mypy, all real MetaDrive scenario smokes, and repeatability checks.

## 12. Delivery sequence

PR #5 is stacked on PR #4 and uses four reviewable implementation commits:

1. Lead Brake plus common Actor manager/config infrastructure;
2. Cut-in;
3. Occluded Crossing Actor;
4. Curriculum, durable logging, resume state, and final smoke verification.

Phase 6 remains a separate future branch and PR.
