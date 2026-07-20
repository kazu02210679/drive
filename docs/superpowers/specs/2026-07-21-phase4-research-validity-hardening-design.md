# Phase 4.1 Research-Validity Hardening Design

## 1. Purpose

Phase 4 can run PPO against MetaDrive, but its current episode seeding, simulator-state
boundary, scenario lifecycle, reward inputs, failure handling, and artifact ownership are
not strong enough for research comparisons. Phase 4.1 hardens those boundaries before
the three Phase 5 scenarios are added.

The resulting system must preserve the existing 24-dimensional Coordinator observation
shape. Explicit observation validity flags and any increase from 24 dimensions are a
separate follow-up. Existing checkpoints remain structurally loadable by raw SB3 because
the spaces do not change, but the project CLI rejects pre-hardening checkpoints that lack
compatible research-contract metadata. Models trained before this hardening must not be
mixed with models trained after it in a formal result.

## 2. Scope

Phase 4.1 includes:

1. reproducible episode-seed sequences that advance on implicit Gymnasium resets;
2. separate episode RNG, MetaDrive scenario, and scenario-parameter seed identities;
3. disjoint train, validation, and test seed ranges;
4. a scenario runtime lifecycle boundary used by Phase 5;
5. an agent-visible observation separated from privileged simulator truth;
6. current-state reward fixes for unnecessary braking and off-road events;
7. per-agent failure isolation;
8. fail-fast handling for internal environment faults;
9. strict run-directory ownership and resume provenance;
10. explicit simulation timing and coordinate contracts;
11. corrected PPO and comparison-protocol requirements;
12. multiple RiskClaims per specialist with deterministic 24-dimensional aggregation.

Phase 4.1 does not implement Lead Brake, Cut-in, or Occluded Crossing actors. It creates
the interfaces those scenarios require. It does not add observation validity slots,
curriculum progression, Phase 6 reports, plots, GIFs, or final benchmark execution.

## 3. Terminology

The decision system contains three specialist agents—Nominal, Hazard, and Rule—and one
Critic. The Critic reviews specialist claims; it is not a fourth specialist.

An episode has three distinct seed identities:

- `episode_rng_seed`: initializes the outer Gymnasium episode RNG.
- `metadrive_scenario_index`: selects MetaDrive's road/traffic scenario.
- `scenario_parameter_seed`: generates scenario-specific parameters such as event time,
  gap, actor speed, and deceleration.

`EnvironmentRole` is one of `train`, `validation`, or `test`. Validation is used by
`EvalCallback`, curriculum decisions, and best-checkpoint selection. Test is reserved for
Phase 6 final comparisons and is never consumed during training or model selection.

## 4. Configuration Contracts

### 4.1 Simulation timing

`MetaDriveConfig` gains:

```yaml
metadrive:
  physics_dt_s: 0.02
  decision_repeat: 5
  decision_dt_s: 0.10
  lane_width_m: 3.5
```

All values are finite and positive; `decision_repeat` is a positive integer. `lane_width_m`
is the fallback lane width for geometric same-lane checks and maps to MetaDrive's
`map_config.lane_width`. Configuration
validation requires, within floating-point tolerance:

```text
decision_dt_s == physics_dt_s * decision_repeat
```

The adapter maps the first two values to MetaDrive's `physics_world_step_size` and
`decision_repeat`. Runtime state must match the validated values. TTC, reaction delay,
PID integration, jerk, standstill cost, scenario event time, and claim validity all use
`decision_dt_s` from this single contract.

### 4.2 Seed splits

The initial research protocol uses non-overlapping defaults:

```yaml
scenarios:
  train:
    seed_start: 0
    seed_count: 10000
  validation:
    seed_start: 10000
    seed_count: 1000
  test:
    seed_start: 20000
    seed_count: 1000
```

Every range is half-open: `[seed_start, seed_start + seed_count)`. Configuration rejects
empty or overlapping ranges. Environment construction requires an explicit role and
worker index. A worker derives its reproducible seed stream from its role range, worker
index, and configured training seed without crossing into another role's range.

The role-specific factory overrides MetaDrive's `start_seed` and `num_scenarios` with the
selected role's `seed_start` and `seed_count`. Therefore every actual MetaDrive scenario
index belongs to the same disjoint role range as the scenario-parameter seed. The old
single global MetaDrive seed range is not used by training factories after this change.

Formal PPO comparisons use at least five training seeds: `42, 43, 44, 45, 46`. These are
policy/RNG seeds, not replacements for the disjoint scenario ranges.

### 4.3 Reward and failure settings

`unnecessary_brake_lookahead_steps` is removed. The learning reward uses only the current
post-step state. "No event occurred later" remains an evaluation metric, not a training
reward input.

Internal simulator, snapshot, reward, or observation construction errors are programming
or infrastructure faults, not MDP terminal states. Phase 4.1 therefore uses fail-fast
behavior and propagates the exception after closing owned simulator resources. It does not
emit a zero observation or encode the fault as `truncated=True`.

## 5. Episode Seed Lifecycle

The first reset establishes the Gymnasium RNG:

- `reset(seed=N)` reinitializes the RNG with `N` and uses `N` as that episode's RNG seed.
- If the first call is `reset()` without a seed, the configured application seed initializes
  the RNG and is used for the first episode.
- A later implicit `reset()` draws the next episode RNG seed from `self.np_random`.

Consequently, two environments started with the same explicit seed produce the same
sequence, while consecutive implicit resets within one environment produce different
episode seeds. An explicit reset restarts the sequence.

The role-specific allocator uses
`numpy.random.SeedSequence([episode_rng_seed, role_code, worker_index])`, where role codes
are `train=0`, `validation=1`, and `test=2`. It spawns two child sequences. The first child
generates an unsigned integer that is reduced modulo `seed_count` and offset by
`seed_start` to produce `metadrive_scenario_index`; the second does the same independently
for `scenario_parameter_seed`. This exact derivation is deterministic and keeps both values
inside the role range. Equal numeric values are allowed because the identities serve
different consumers.

MetaDrive scenario selection is explicit and separately logged. The adapter must return the
actual MetaDrive index; the outer environment never reports an unwrapped requested seed as
the actual road scenario. A mismatch between requested and returned scenario index is an
environment error.

Reset `info`, every decision trace, and training metadata record all three seed identities
plus environment role and worker index.

## 6. Scenario Runtime Boundary

Phase 5 scenarios plug into the environment through a simulator-independent protocol:

```python
class ScenarioRuntime(Protocol):
    def reset(
        self,
        env: DrivingEnvironment,
        *,
        seeds: EpisodeSeeds,
    ) -> ScenarioState: ...

    def after_simulator_reset(
        self,
        env: DrivingEnvironment,
        state: ScenarioState,
    ) -> None: ...

    def before_step(
        self,
        env: DrivingEnvironment,
        state: ScenarioState,
        *,
        step_index: int,
    ) -> None: ...

    def after_step(
        self,
        env: DrivingEnvironment,
        state: ScenarioState,
        *,
        step_index: int,
        raw_info: Mapping[str, object],
    ) -> ScenarioStepResult: ...

    def observation_context(
        self,
        state: ScenarioState,
    ) -> ScenarioObservationContext: ...
```

The environment calls these hooks in this order:

1. allocate episode seeds and select the scenario;
2. call `ScenarioRuntime.reset` before simulator reset;
3. reset MetaDrive with the selected scenario index;
4. call `after_simulator_reset` for actor creation and policy assignment;
5. build the initial scene frame and agent analysis;
6. before each simulator step, call `before_step` for event triggers;
7. advance MetaDrive;
8. call `after_step` for scenario status and labels;
9. build the next frame, claims, reward, observation, and trace.

The Phase 4.1 default runtime is a deterministic no-op implementation. Scenario-specific
logic is not embedded directly in `MultiAgentSpeedEnv`.

`ScenarioObservationContext` carries scenario ID, stop requirement, occlusion regions,
distance to conflict point, intersection-entry prohibition, and visible actor IDs. If an
occlusion is active but visibility metadata is absent, frame construction fails. This
fail-closed rule prevents accidental truth leakage.

## 7. Observation and Privileged-State Boundary

The current all-purpose snapshot is replaced by three immutable types:

```python
@dataclass(frozen=True)
class SceneObservation:
    step_index: int
    sim_time_s: float
    scenario_id: str
    seeds: EpisodeSeeds
    ego: EgoState
    visible_actors: tuple[ActorState, ...]
    occlusion_regions: tuple[OcclusionRegion, ...]
    road_context: RoadContext
    previous_executed_action: int
    previous_shield_intervention: bool

@dataclass(frozen=True)
class PrivilegedWorldState:
    all_actors: tuple[ActorState, ...]
    collision_occurred: bool
    collision_kind: CollisionKind | None
    off_road: bool
    arrived: bool
    scenario_success: bool
    scenario_failure: bool

@dataclass(frozen=True)
class SceneFrame:
    observation: SceneObservation
    privileged: PrivilegedWorldState
```

Specialists, Critic, Coordinator, and Safety Shield receive only `SceneObservation`.
Reward, evaluation, and debug logging may receive `PrivilegedWorldState`. Hidden actors do
not appear in `visible_actors`; setting `visible=False` while retaining their kinematics in
the agent-facing structure is forbidden.

Current ego and road facts required for action safety remain observable. Collision kind,
arrival, scenario outcome, and hidden-actor state are privileged. Off-road status is built
from `raw_info["out_of_road"] OR vehicle.on_lane-derived state` and is consumed through the
privileged reward boundary.

## 8. Coordinate Contract

- `position_xy_m` and `velocity_xy_mps` use MetaDrive world XY coordinates in metres and
  metres per second.
- `heading_rad` is MetaDrive's world heading in radians, normalized to `[-pi, pi)` at the
  project boundary; positive rotation is counter-clockwise.
- `relative_longitudinal_m` and `relative_lateral_m` use the ego body frame. Longitudinal is
  positive forward. Lateral is positive to the ego's left.
- `lane_offset_m` uses the current lane's local coordinate system and preserves MetaDrive's
  lateral sign. Tests pin the observed sign against a known straight-lane fixture.
- `same_lane` requires equal canonical lane indices and an actor position inside the lane
  width. Equal indices alone are insufficient.
- Each scenario runtime computes its conflict point from scenario geometry and supplies the
  ego-path distance in metres. The snapshot builder does not infer scenario conflict
  geometry.

## 9. Multiple Claims and Failure Isolation

Each specialist implements:

```python
def analyze(observation: SceneObservation) -> tuple[RiskClaim, ...]: ...
```

It returns one to three claims in deterministic safety order. Candidate ordering is:

1. `hard_stop_required=True` first;
2. higher severity;
3. lower finite TTC, with missing TTC after finite TTC;
4. lower finite stopping margin;
5. lower recommended maximum speed;
6. ascending target actor ID and event type as stable tie-breakers.

`AgentSuite` catches exceptions around each specialist separately. It preserves successful
claims and returns an `AgentAnalysisResult` containing claims, failed agent IDs, and
sanitized error summaries. Runtime failure is distinct from an intentional ablation. The
Critic sees all surviving claims and the failed-agent identities. Safety Shield treats a
failed specialist as missing according to its existing conservative policy.

Critic and Shield consume every claim. DecisionTrace stores every claim. The fixed
24-dimensional Coordinator observation receives one aggregate per specialist:

- minimum finite TTC;
- minimum finite stopping margin;
- maximum severity and probability;
- minimum recommended maximum speed;
- any hard-stop requirement;
- minimum confidence.

Aggregation is field-wise and safety-conservative; it does not select one target and discard
other fields. Existing observation slots retain their positions. Claim count, target count,
explicit validity, and explicit agent-failure slots are deferred to the observation-schema
follow-up.

## 10. Reward Semantics

The ten existing reward component names remain stable.

Unnecessary braking is penalized in the current post-step state when all conditions hold:

- the executed action is `SLOW`, `PREPARE_STOP`, or `STOP`;
- Hazard and Rule completed successfully;
- every Hazard claim is below the configured severity threshold;
- TTC is missing because there is no hazard actor, or minimum TTC is at least the safe TTC;
- no current hard rule constraint exists;
- no collision, off-road event, or Shield intervention occurred.

The penalty applies immediately as `-scale * executed_action`. No future observation and no
safe-brake streak is used.

Collision, arrival, off-road, and scenario outcome inputs come from privileged post-step
state. Near-miss TTC comes from current specialist claims and is therefore based on data
available at the decision boundary.

## 11. Control and PPO Contracts

`STOP` already resets the speed PID and uses configured emergency deceleration. Phase 4.1
retains this behavior and adds an end-to-end test proving that Shield-enforced STOP reaches
the emergency path. `SLOW` and `PREPARE_STOP` are interpreted as speed caps; STOP is both a
zero speed cap and an emergency-brake request.

The outer Gymnasium environment owns scenario logic, specialists, Critic, Coordinator,
Shield, reward, and traces. The inner MetaDrive policy owns only lane keeping and conversion
from the selected high-level speed action to steering/throttle/brake.

Training uses standard Stable-Baselines3 PPO. Updates occur after `n_steps * num_envs`
rollout transitions and are not synchronized to episode boundaries. The specification's
old prohibition on mid-episode updates is removed; no custom rollout collector is added.

## 12. Internal Errors and Gymnasium Semantics

Natural MDP outcomes use Gymnasium semantics:

- collision, scenario-defined failure, or success: `terminated=True`;
- configured time horizon: `truncated=True`;
- internal simulator or project error: close owned resources and raise the original error.

Internal errors never return an all-zero observation, reward zero, or a false time-limit
transition. This prevents PPO from bootstrapping a software fault as though it were a valid
truncated episode.

Per-agent analysis failure is not an environment failure. It follows the isolated fallback
path in Section 9 and remains visible in trace metadata.

## 13. Training Artifact Ownership and Resume

New training requires a nonexistent or empty run directory. A non-empty directory is
rejected before config, TensorBoard events, or checkpoints are written. Phase 4.1 does not
add an overwrite option; users choose a new run directory explicitly.

Resume also writes to a new empty run directory. It records:

- SHA-256 of the parent checkpoint;
- resolved parent checkpoint path and parent run directory when discoverable;
- current resolved config;
- parent config when present;
- a machine-readable parent/current config difference;
- starting `model.num_timesteps`;
- observation shape and action count.

Every run records `research_contract_version=2` and `observation_schema_version=1`.
Pre-hardening checkpoints have no compatible research-contract metadata and are rejected by
the project resume command, even though raw SB3 could load their unchanged spaces.

After loading, `policy`, `learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`,
`gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm`, observation shape, and
action count are compared to the current config. A mismatch fails before learning.
`total_timesteps`, smoke timesteps, callback frequencies, run root, and the new training
seed may differ because they control the continuation run rather than the loaded optimizer
contract. Operational output never claims the current config was used when the loaded
checkpoint retains incompatible values.

Checkpoint promotion remains transactional. Existing historical run directories and
checkpoints are never deleted by Phase 4.1.

## 14. Comparison Protocol

Phase 6 must report two separate comparisons:

1. Decision performance: B1, B2, and Proposed all use `shield=monitor`. Shield diagnostics
   are logged but actions are not changed.
2. Executable system performance: every compared method uses `shield=enforce`. Reports
   include collision rate, raw unsafe-request rate, intervention rate, and post-Shield
   outcomes.

`Proposed without Shield` is an ablation, not the main baseline. B0, B1, B2, and Proposed
use the same final test scenario seeds within each comparison. Final test seeds are never
used for checkpoint selection or curriculum progression.

## 15. Verification Strategy

Every behavior change follows red-green-refactor TDD. Required focused tests include:

- explicit reset reproducibility, advancing implicit reset seeds, and cross-environment
  seed-sequence reproducibility;
- VecEnv automatic reset advancing to a new episode seed;
- distinct and correctly logged episode, MetaDrive, and parameter seeds;
- rejection of overlapping train/validation/test ranges;
- exact ScenarioRuntime hook order and no-op compatibility;
- hidden actors absent from every agent-facing object and present only in privileged state;
- active occlusion without visibility metadata failing closed;
- coordinate signs and same-lane geometry;
- one specialist failing while other claims survive;
- multi-claim deterministic ordering and field-wise 24-slot aggregation;
- no-actor unnecessary braking, current hazard, current rule constraint, and Shield cases;
- `raw_info["out_of_road"]` contributing the off-road penalty;
- simulator, snapshot, reward, and observation errors propagating instead of returning a
  truncated transition;
- non-empty run-directory rejection before writes;
- resume SHA-256/provenance metadata and hyperparameter mismatch rejection;
- Shield-enforced STOP reaching emergency deceleration;
- simulation timing configuration and runtime mismatch rejection.

After focused tests, verification runs the complete unit and integration suite, Ruff,
mypy, coverage, a real headless MetaDrive smoke episode, and a PPO smoke-training run in a
fresh run directory. The PPO smoke check must observe at least two distinct episode seeds
while reproducing the same sequence in a second run started with the same training seed.

## 16. Documentation and Migration

The top-level MVP specification, README, example YAML files, Phase 4 implementation log,
and CLI help are updated with the new contracts. The implementation log records Phase 4.1
as research-validity hardening rather than claiming that old smoke results are comparable
to post-hardening experiments.

The observation remains 24-dimensional. A follow-up issue/design must add explicit
`ttc_valid`, `claim_valid`, `agent_failed`, and `target_actor_present` features before final
Level 3 multi-hazard experiments. That follow-up requires retraining and a versioned
observation schema.
