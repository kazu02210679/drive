# Phase 6 Evaluation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Phase 6 evaluation pipeline that compares the specified baselines, proposed method, and ablations on matched test episodes, publishes strict auditable artifacts, and updates the architecture-map flow cards after implementation.

**Architecture:** A central method-profile registry configures the same environment and agent stack for every method. A deterministic evaluation planner binds methods, checkpoints, policy seeds, scenario cells, and test seeds before execution. The online runner writes strict step and episode records into a staging directory; offline reducers then select checkpoints, aggregate matched comparisons, extract training metrics, render plots/GIFs, and atomically publish a hash-verified evaluation bundle.

**Tech Stack:** Python 3.11, MetaDrive 0.4.3, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, NumPy 1.26.4, Pydantic 2.11.7, pandas 2.3.1, matplotlib 3.10.3, imageio 2.37.0, PyYAML 6.0.2, pytest 8.4.1.

## Global Constraints

- Treat `docs/multi_agent_driving_mvp_spec.md` and `docs/superpowers/specs/2026-07-23-phase6-evaluation-design.md` as the requirements hierarchy.
- Preserve `OBSERVATION_SCHEMA_VERSION = 1`, observation shape `(24,)`, and action order `KEEP`, `SLOW`, `PREPARE_STOP`, `STOP`.
- Increment `RESEARCH_CONTRACT_VERSION` from 6 to 7 when method-profile provenance becomes mandatory.
- Keep train `[0, 10000)`, validation `[10000, 11000)`, and test `[20000, 21000)` disjoint. Never use test results for training, curriculum progression, or checkpoint selection.
- Compute reward and evaluation safety metrics from privileged simulator truth, never from the compared Agent Claims.
- Give every compared method the same physical transition, scenario parameters, test seed, horizon, reward function, and metric reducer within a matched comparison cell.
- Treat an intentionally omitted specialist as an ablation, not an Agent failure. Preserve runtime-failure handling for expected specialists.
- Give B0 only the current immutable `SceneObservation`; it must not inspect privileged state, future frames, or another method's output.
- Do not duplicate deterministic B0 as five pseudo policy seeds. Its `policy_seed` is `None`.
- Keep no-shield ablation in `shield.mode=off`; use `monitor` for the decision track and `enforce` for the executable-system track.
- Generate plots, GIFs, and reports from persisted records after simulation. Offline commands must not import or launch MetaDrive.
- Reject malformed, incomplete, mixed-contract, mixed-split, duplicate, or non-finite records instead of silently dropping them.
- Write evaluation output to a sibling staging directory and publish with an atomic rename only after validation and SHA-256 manifest generation.
- The first implementation run is a small end-to-end smoke comparison. The formal five-policy-seed research run is explicitly deferred.
- Every production change follows red-green-refactor TDD, with the failing test observed before implementation.
- Final gates: Ruff, Ruff format check, strict mypy, complete pytest with branch coverage at least 90%, deterministic fake-pipeline replay, real headless MetaDrive smoke, and architecture-map validation.

## File Map

### Method profiles and runtime provenance

- Create `src/mad_driving/methods.py`.
- Create `src/mad_driving/agents/noop_critic.py`.
- Modify `src/mad_driving/agents/suite.py` and `src/mad_driving/agents/__init__.py`.
- Modify `src/mad_driving/config/models.py` and `configs/base.yaml`.
- Create seven overlays under `configs/methods/`.
- Modify `src/mad_driving/envs/multi_agent_speed_env.py`.
- Modify `src/mad_driving/training/metadata.py` and its construction sites.
- Modify method/config/environment/metadata unit tests.

### Oracle telemetry and policy execution

- Modify `src/mad_driving/interfaces/scene_frame.py`.
- Modify `src/mad_driving/interfaces/decision_trace.py`.
- Modify `src/mad_driving/world_model/snapshot_builder.py`.
- Modify `src/mad_driving/envs/multi_agent_speed_env.py`.
- Create `src/mad_driving/evaluation/__init__.py` and `policies.py`.
- Add unit tests for oracle stopping margin, B0, PPO adaptation, and latency/trace data.

### Planning, persistence, execution, and metrics

- Create `src/mad_driving/evaluation/models.py`.
- Create `src/mad_driving/evaluation/plans.py`.
- Create `src/mad_driving/evaluation/serialization.py`.
- Create `src/mad_driving/evaluation/workspace.py`.
- Create `src/mad_driving/evaluation/runner.py`.
- Create `src/mad_driving/evaluation/metrics.py`.
- Create `src/mad_driving/evaluation/selection.py`.
- Create `src/mad_driving/evaluation/compare.py`.
- Create corresponding unit tests under `tests/unit/evaluation/`.

### Offline analysis and command line

- Create `src/mad_driving/evaluation/training_metrics.py`.
- Create `src/mad_driving/visualization/__init__.py`, `plots.py`, `overlay.py`, and `report.py`.
- Create `src/mad_driving/cli/evaluate.py`, `compare.py`, and `render_episode.py`.
- Create `configs/evaluation/phase6_smoke.yaml`.
- Create CLI, visualization, fake end-to-end, and real MetaDrive integration tests.
- Modify `README.md` and `docs/multi_agent_driving_mvp_spec.md` only to document the implemented commands/artifact contract; do not change the scientific requirement.

### Architecture-map follow-up

- In the separate workspace `C:\Users\楫屋寿弥\Documents\Codex\2026-07-22\kazu02210679-drive-git\work\drive`, update `architecture-map.json` and `architecture-map.html` on `feat-architecture-map-flow-cards`.
- Run `tools/validate_architecture_map.py` and any existing architecture-map QA command.
- Commit and push that branch only after Phase 6 behavior and artifact names are final.

---

### Task 1: Add the central method-profile contract

**Files:**
- Create `src/mad_driving/methods.py`.
- Create `src/mad_driving/agents/noop_critic.py`.
- Modify `src/mad_driving/agents/suite.py` and `src/mad_driving/agents/__init__.py`.
- Modify `src/mad_driving/config/models.py` and `configs/base.yaml`.
- Create `configs/methods/b0_rule.yaml`, `b1_nominal.yaml`, `b2_multi_no_review.yaml`, `proposed.yaml`, `proposed_no_critic.yaml`, `proposed_no_shield.yaml`, and `proposed_no_hazard.yaml`.
- Create `tests/unit/test_methods.py` and modify config/Agent-suite tests.

**Interfaces:**

```python
# `MethodId` and `MethodConfig` live in config/models.py. methods.py imports and
# re-exports MethodId, avoiding a config.models <-> methods import cycle.
MethodId = Literal[
    "b0_rule",
    "b1_nominal",
    "b2_multi_no_review",
    "proposed",
    "proposed_no_critic",
    "proposed_no_shield",
    "proposed_no_hazard",
]

@dataclass(frozen=True)
class MethodProfile:
    method_id: MethodId
    policy_kind: Literal["rule", "ppo"]
    specialist_ids: tuple[Literal["nominal", "hazard", "rule"], ...]
    critic_enabled: bool
    default_shield_mode: Literal["off", "monitor", "enforce"]

def get_method_profile(method_id: MethodId) -> MethodProfile: ...
def build_method_suite(config: AgentsConfig, method_id: MethodId) -> AgentSuite: ...
```

- [ ] **Step 1: Write failing registry and strict-overlay tests**

Test the exact seven IDs, specialist sets, policy kind, critic flag, and default shield mode. Assert B0 expects no specialists and uses a no-op reviewer, B1 expects only `nominal`, no-hazard expects `nominal` and `rule`, B2 uses all three specialists with a no-op reviewer, proposed uses the real Critic, and no-shield defaults to `off`. Load every method overlay and reject unknown IDs/keys.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_methods.py tests/unit/config -q`

Expected: FAIL because `MethodId`, `MethodProfile`, method config, overlays, and profile-aware suite construction do not exist.

- [ ] **Step 3: Implement immutable profiles and intentional no-op review**

Define `MethodId` and `MethodConfig(id: MethodId = "proposed")` in `config/models.py`, nest the latter in `AppConfig`, and re-export `MethodId` from `methods.py`. This direction keeps `methods.py -> config.models` and prevents a circular import. Keep one exhaustive `_METHOD_PROFILES` mapping and assert at import/test time that its keys equal the `MethodId` literal values. `NoOpCritic.review(...)` must return a neutral immutable `CriticReview` with reason `critic_intentionally_disabled`; it must not rewrite actions or fabricate risk.

`build_method_suite` must instantiate only configured specialists. A missing profile specialist must be represented as `None`, so `AgentSuite.expected_agent_ids` remains the source of truth for intentional omissions versus failures.

- [ ] **Step 4: Run method/config tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_methods.py tests/unit/config tests/unit/agents -q`

Expected: PASS.

- [ ] **Step 5: Commit the method-profile slice**

Run:

```powershell
git add src/mad_driving/methods.py src/mad_driving/agents src/mad_driving/config/models.py configs/base.yaml configs/methods tests/unit
git commit -m "feat: add phase 6 method profiles"
```

---

### Task 2: Bind profiles to the environment and research provenance

**Files:**
- Modify `src/mad_driving/envs/multi_agent_speed_env.py`.
- Modify `src/mad_driving/training/metadata.py` and training metadata construction sites.
- Modify `tests/unit/envs/test_multi_agent_speed_env.py`, `tests/unit/training/test_metadata.py`, and `tests/unit/training/test_train.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class MethodProfileSnapshot:
    method_id: str
    policy_kind: str
    specialist_ids: tuple[str, ...]
    critic_enabled: bool
    shield_mode: str

class RunMetadata:
    method_profile: MethodProfileSnapshot
    research_contract_version: int = 7
```

- [ ] **Step 1: Write failing environment-composition tests**

Construct the environment with each profile and assert its `expected_agent_ids`, Critic implementation, and active shield mode. Inject an Agent runtime failure and prove that expected-but-failed agents still reach the Shield, while intentionally absent agents do not.

- [ ] **Step 2: Write failing provenance tests**

Assert metadata records the complete profile snapshot, refuses a snapshot inconsistent with resolved config, and rejects research contract version 6 when loading a Phase 6 run. Preserve observation/action schema versions at 1.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/envs/test_multi_agent_speed_env.py tests/unit/training/test_metadata.py tests/unit/training/test_train.py -q`

Expected: FAIL because default environment composition ignores `config.method`, metadata lacks a method profile, and the research contract is still 6.

- [ ] **Step 4: Implement profile-aware defaults and v7 metadata**

Make `suite_factory` optional. If no test/integration factory is injected, build from `config.method.id`; preserve all existing injection seams. Resolve the profile once at run construction, serialize it into metadata, and validate profile/config/checkpoint agreement when starting or resuming training/evaluation.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/envs/test_multi_agent_speed_env.py tests/unit/training/test_metadata.py tests/unit/training/test_train.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the runtime/provenance slice**

Run:

```powershell
git add src/mad_driving/envs/multi_agent_speed_env.py src/mad_driving/training tests/unit/envs tests/unit/training
git commit -m "feat: bind method profiles to runtime provenance"
```

---

### Task 3: Add method-independent stopping-margin truth and timing telemetry

**Files:**
- Modify `src/mad_driving/interfaces/scene_frame.py` and `decision_trace.py`.
- Modify `src/mad_driving/world_model/snapshot_builder.py`.
- Modify `src/mad_driving/envs/multi_agent_speed_env.py`.
- Modify matching interface, world-model, and environment tests.

**Interfaces:**

```python
class PrivilegedWorldState:
    minimum_actual_stopping_margin_m: float | None

class DecisionTrace:
    expected_agent_ids: tuple[str, ...]
    analysis_latency_ms: float
    shield_latency_ms: float

def stopping_margin_m(
    *,
    ego_speed_mps: float,
    minimum_actual_ttc_s: float | None,
    reaction_delay_s: float,
    safe_deceleration_mps2: float,
) -> float | None: ...
```

The exact oracle formula is:

```python
available_distance_m = ego_speed_mps * minimum_actual_ttc_s
required_distance_m = (
    ego_speed_mps * reaction_delay_s
    + ego_speed_mps**2 / (2.0 * safe_deceleration_mps2)
)
margin_m = available_distance_m - required_distance_m
```

- [ ] **Step 1: Write failing stopping-margin tests**

Cover no TTC, zero speed, positive margin, negative margin, invalid non-positive safe deceleration, and hidden-Actor truth. Prove the result is identical when Claims and enabled specialists differ.

- [ ] **Step 2: Write failing trace timing tests**

Patch the monotonic nanosecond clock with deterministic values. Assert finite non-negative millisecond durations and exact expected Agent IDs. Confirm timing fields are excluded from any deterministic scientific-equality helper because wall-clock values are nondeterministic.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/interfaces tests/unit/world_model tests/unit/envs/test_multi_agent_speed_env.py -q`

Expected: FAIL because the oracle field and latency/expected-agent trace fields do not exist.

- [ ] **Step 4: Implement the oracle and instrumentation**

Compute stopping margin in the snapshot builder from privileged TTC and configured Hazard reaction/deceleration constants. Measure analysis and Shield calls with `time.perf_counter_ns()`, convert once to milliseconds, and validate finiteness. Do not include JSON serialization, simulator stepping, rendering, or reward calculation in those two durations.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/interfaces tests/unit/world_model tests/unit/envs/test_multi_agent_speed_env.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the oracle/telemetry slice**

Run:

```powershell
git add src/mad_driving/interfaces src/mad_driving/world_model src/mad_driving/envs tests/unit/interfaces tests/unit/world_model tests/unit/envs
git commit -m "feat: add evaluation oracle and timing telemetry"
```

---

### Task 4: Implement strict evaluation plans and record schemas

**Files:**
- Create `src/mad_driving/evaluation/__init__.py`, `models.py`, `plans.py`, and `serialization.py`.
- Create `tests/unit/evaluation/test_models.py`, `test_plans.py`, and `test_serialization.py`.

**Interfaces:**

```python
EvaluationTrack = Literal["decision", "system", "ablation"]
ScenarioCellId = Literal[
    "level0_nominal",
    "level1_lead_brake",
    "level2_lead_brake",
    "level2_cut_in",
    "level3_occluded_crossing",
]

@dataclass(frozen=True)
class EvaluationRunSpec:
    track: EvaluationTrack
    method_id: MethodId
    policy_seed: int | None
    checkpoint_path: str | None
    scenario_cell_id: ScenarioCellId
    episode_index: int
    test_seed: int
    shield_mode: Literal["off", "monitor", "enforce"]

@dataclass(frozen=True)
class EvaluationStepRecord: ...

@dataclass(frozen=True)
class EvaluationEpisodeRecord: ...

class PpoRunBinding(StrictFrozenModel):
    method_id: MethodId
    policy_seed: int
    training_run_dir: Path

class EvaluationPlanConfig(StrictFrozenModel):
    plan_kind: Literal["phase6_smoke", "phase6_formal"]
    evaluation_id: str
    app_config_path: Path
    episodes_per_case: PositiveInt
    test_seed_start: int
    ppo_run_bindings: tuple[PpoRunBinding, ...]
    capture_episode_keys: tuple[str, ...]

def load_evaluation_plan(path: Path) -> EvaluationPlanConfig: ...
def build_smoke_plan(...) -> tuple[EvaluationRunSpec, ...]: ...
def build_formal_plan(...) -> tuple[EvaluationRunSpec, ...]: ...
def write_jsonl_strict(path: Path, rows: Iterable[Mapping[str, object]]) -> None: ...
def read_jsonl_strict(path: Path, model: type[T]) -> tuple[T, ...]: ...
```

- [ ] **Step 1: Write failing fixed-matrix plan tests**

Assert the exact tracks:

- decision: B1, B2, proposed with `monitor`;
- system: B0, B1, B2, proposed with `enforce`;
- ablation: proposed, no-Critic, no-Shield, no-Hazard, with only no-Shield set to `off`.

Assert all five scenario cells are represented, test seeds are matched within each comparison cell, B0 has `policy_seed=None` and no checkpoint, PPO rows bind an explicit seed/checkpoint, ordering is stable, and duplicate keys are rejected.
Assert a formal plan uses exactly policy seeds `(42, 43, 44, 45, 46)` for every PPO method, while the smoke plan uses one explicitly bound PPO seed and is marked non-formal.

- [ ] **Step 2: Write failing strict-record tests**

Require schema version, contract version, track/method/profile/checkpoint identity, all episode seeds, scenario cell, step index, observation/action fields, oracle fields, all ten reward components, Claim/review/Shield fields, policy-inference/Agent-analysis/Shield/total-decision latency fields, termination flags, and frame reference. Reject NaN/Infinity, extra keys, missing keys, inconsistent totals, and success/failure contradictions.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_models.py tests/unit/evaluation/test_plans.py tests/unit/evaluation/test_serialization.py -q`

Expected: FAIL because the evaluation package does not exist.

- [ ] **Step 4: Implement immutable records and deterministic planners**

Use dataclasses with complete `__post_init__` validation and explicit `to_dict`/`from_dict` conversions. Parse the plan YAML with a strict frozen Pydantic model and reject unknown keys. Use sorted-key UTF-8 JSON, one object per line, and refuse overwrite by default. Plan generation must be a pure function and must not probe the filesystem; the CLI/orchestration boundary validates referenced training runs and checkpoints.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_models.py tests/unit/evaluation/test_plans.py tests/unit/evaluation/test_serialization.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the planning/schema slice**

Run:

```powershell
git add src/mad_driving/evaluation tests/unit/evaluation
git commit -m "feat: define strict evaluation plans and records"
```

---

### Task 5: Add B0/PPO policy adapters and the online runner

**Files:**
- Create `src/mad_driving/evaluation/policies.py`, `runner.py`, and `workspace.py`.
- Modify `src/mad_driving/evaluation/__init__.py`.
- Modify `src/mad_driving/scenarios/manager.py` and `factory.py`.
- Modify `src/mad_driving/envs/multi_agent_speed_env.py`.
- Modify `src/mad_driving/training/episode_seeds.py` to call the generalized hook.
- Create `tests/unit/evaluation/test_policies.py`, `test_runner.py`, and `test_workspace.py`.
- Modify `tests/unit/scenarios/test_manager.py` and `tests/unit/envs/test_multi_agent_speed_env.py`.

**Interfaces:**

```python
class EvaluationPolicy(Protocol):
    def reset(self) -> None: ...
    def predict(self, observation: SceneObservation | np.ndarray) -> int: ...

class VisibleTtcRulePolicy:
    STOP_TTC_S: Final[float] = 1.0
    PREPARE_STOP_TTC_S: Final[float] = 3.0
    SLOW_TTC_S: Final[float] = 5.0

class PpoPolicyAdapter: ...

@dataclass(frozen=True)
class EvaluationRunResult:
    step_records: tuple[EvaluationStepRecord, ...]
    episode_record: EvaluationEpisodeRecord

def run_evaluation_episode(spec: EvaluationRunSpec, ...) -> EvaluationRunResult: ...

class EvaluationWorkspace:
    @classmethod
    def stage(cls, destination: Path) -> EvaluationWorkspace: ...
    def publish(self) -> Path: ...

class MultiAgentSpeedEnv:
    def set_evaluation_scenario_schedule(self, scenario_ids: tuple[str, ...]) -> None: ...
    def current_scene_observation_for_evaluation(self) -> SceneObservation: ...
```

- [ ] **Step 1: Write failing finite evaluation-schedule tests**

Install explicit schedules on validation and test environments, select `lead_brake` and `cut_in` independently at Level 2, consume entries once in order, and fail when the schedule is exhausted. Assert training environments reject schedule installation, scheduled scenarios outside the pending difficulty level are rejected, and the reset's explicit episode seed still derives all four recorded scenario seeds.

- [ ] **Step 2: Write failing B0 tests**

Test the fixed visible TTC boundaries at `1.0`, `3.0`, and `5.0` seconds and deterministic action selection. Include a visible stop requirement, prohibited intersection entry, same-lane Actors, intersecting trajectories, and occluded/hidden Actors. Prove B0 can obtain only the environment's current immutable `SceneObservation` through the evaluation-only read method and cannot access privileged Actors. Assert `STOP`, `PREPARE_STOP`, `SLOW`, and `KEEP` exactly as specified; do not add user-overridable B0 thresholds.

- [ ] **Step 3: Write failing PPO adapter tests**

Use a fake SB3 model to assert deterministic prediction, scalar integer conversion, action-range validation, reset behavior, and rejection of observation/schema/checkpoint metadata mismatches.

- [ ] **Step 4: Write failing runner/workspace tests**

With a fake Gymnasium environment, assert reset options carry the fixed test seed and scenario cell, every step is logged exactly once, terminal/truncated handling is correct, the episode record matches step totals, and a rerun produces byte-identical scientific records after removing latency. Assert partial failure leaves only staging data and never a published destination. Assert existing destinations are not overwritten.

- [ ] **Step 5: Run focused tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_policies.py tests/unit/evaluation/test_runner.py tests/unit/evaluation/test_workspace.py tests/unit/scenarios/test_manager.py tests/unit/envs/test_multi_agent_speed_env.py -q`

Expected: FAIL because policy adapters, runner, and atomic workspace do not exist.

- [ ] **Step 6: Implement the finite schedule, adapters, runner, and atomic staging**

Generalize the validation-only schedule hook to `set_evaluation_scenario_schedule`, permit only `validation` and `test` roles, update the curriculum wrapper call site, and preserve one finite ordered schedule per runtime factory. The runner installs a one-entry schedule and the planned difficulty level before reset, then verifies returned scenario identity and all derived seeds against the plan. Do not let a scenario sampler override a scheduled case.

The runner must:

1. validate method/profile/checkpoint/config agreement;
2. reset the selected policy and environment with the planned test episode;
3. call the policy using only its permitted observation;
4. write step records incrementally to staging;
5. derive the terminal episode record from persisted steps;
6. close simulator resources in `finally`;
7. fsync/close files before publication;
8. leave the destination absent on any exception.

The B0 adapter consumes `SceneObservation`; PPO consumes the existing 24-vector generated by the environment. Do not add an image-observation path.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_policies.py tests/unit/evaluation/test_runner.py tests/unit/evaluation/test_workspace.py tests/unit/scenarios/test_manager.py tests/unit/envs/test_multi_agent_speed_env.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the online-runner slice**

Run:

```powershell
git add src/mad_driving/evaluation src/mad_driving/scenarios src/mad_driving/envs src/mad_driving/training/episode_seeds.py tests/unit/evaluation tests/unit/scenarios tests/unit/envs tests/unit/training
git commit -m "feat: run reproducible evaluation episodes"
```

---

### Task 6: Reduce episode metrics and select checkpoints on fixed validation

**Files:**
- Create `src/mad_driving/evaluation/metrics.py` and `selection.py`.
- Create `tests/unit/evaluation/test_metrics.py` and `test_selection.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class EpisodeMetrics:
    collision: bool
    crossing_actor_collision: bool
    near_miss: bool
    minimum_actual_ttc_s: float | None
    negative_stopping_margin: bool
    minimum_stopping_margin_m: float | None
    hard_rule_violation: bool
    raw_unsafe_request_rate: float
    shield_intervention_rate: float
    off_road: bool
    scenario_success: bool
    final_route_completion: float
    average_speed_mps: float
    simulated_travel_time_s: float
    unnecessary_braking_event_count: int
    unnecessary_stop_duration_s: float
    longitudinal_acceleration_rms_mps2: float
    maximum_deceleration_mps2: float
    longitudinal_jerk_rms_mps3: float | None
    agent_disagreement_eligible_steps: int
    agent_disagreement_count: int
    agent_disagreement_rate: float | None
    critic_challenge_eligible_steps: int
    critic_challenge_count: int
    critic_challenge_rate: float | None
    critic_found_missed_danger_count: int
    critic_found_missed_danger_rate: float | None
    critic_false_challenge_count: int
    critic_false_challenge_rate: float | None
    agent_failure_fallback_count: int
    decision_latency_p50_ms: float
    decision_latency_p95_ms: float
    decision_latency_p99_ms: float
    episode_reward: float

def reduce_episode(records: Sequence[EvaluationStepRecord], decision_dt_s: float) -> EpisodeMetrics: ...

@dataclass(frozen=True)
class CheckpointScore: ...

def discover_checkpoint_candidates(completed_run_dir: Path) -> tuple[CheckpointCandidate, ...]: ...
def select_checkpoint(scores: Sequence[CheckpointScore]) -> CheckpointScore: ...
```

- [ ] **Step 1: Write failing metric-reduction tests**

Build hand-computed trajectories for collision type, TTC threshold crossing, negative stopping margin, hard-rule violation, route completion, time-weighted speed, simulated time, unnecessary-braking event edges, standstill duration, acceleration RMS, maximum deceleration, jerk RMS, unsafe-request/intervention rates, disagreement/review/fallback counts and eligible rates, decision-latency p50/p95/p99, and reward. Test one-step episodes and missing optional TTC. Serialize a rate with no eligible denominator as `None`/an empty CSV cell, never NaN or zero.

- [ ] **Step 2: Write failing checkpoint-selection tests**

Discover candidates only from a completed training run's verified inventory. Include periodic, per-level-best, and final checkpoints when listed; reject unlisted ZIP files, absent/invalid curriculum sidecars, hash mismatches, method/seed/schema mismatches, and incomplete runs. Then use the fixed all-level validation matrix and assert lexicographic ordering:

1. higher mean reward;
2. lower collision rate;
3. higher success rate;
4. higher route progress;
5. lower timestep, preferring the earlier checkpoint;
6. lexicographically smaller SHA-256 as a final deterministic tie-break.

Reject candidates evaluated on different scenario/seed matrices or any test split row.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_metrics.py tests/unit/evaluation/test_selection.py -q`

Expected: FAIL because reducers and fixed-suite selection do not exist.

- [ ] **Step 4: Implement pure reducers and selection**

Keep reducers simulator-independent. A hard-rule violation is a step where the pre-step oracle requires a hard stop and the executed action is below `STOP`. A near miss is a non-collision episode with a finite oracle TTC below the configured threshold. Count an unnecessary-braking event only on a false-to-true edge of the existing oracle condition. Use the approved Agent-disagreement, Critic-found-missed-danger, and false-challenge definitions verbatim from the design document. Derive rates with explicit zero-denominator behavior. Use one documented nearest-rank implementation for p50/p95/p99 so results do not vary with pandas/NumPy defaults. Emit both `model_selection.csv` and `selected_checkpoints.json`; each selected record contains path, SHA-256, method, policy seed, and the full validation-plan hash.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_metrics.py tests/unit/evaluation/test_selection.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the metric/selection slice**

Run:

```powershell
git add src/mad_driving/evaluation tests/unit/evaluation
git commit -m "feat: reduce metrics and select fixed-suite checkpoints"
```

---

### Task 7: Build matched comparison and training-metric extraction

**Files:**
- Create `src/mad_driving/evaluation/compare.py` and `training_metrics.py`.
- Create `tests/unit/evaluation/test_compare.py` and `test_training_metrics.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class ComparisonRow: ...

def validate_matched_episodes(records: Sequence[EvaluationEpisodeRecord]) -> None: ...
def build_comparison_rows(records: Sequence[EvaluationEpisodeRecord]) -> tuple[ComparisonRow, ...]: ...
def write_comparison_csv(path: Path, rows: Sequence[ComparisonRow]) -> None: ...

@dataclass(frozen=True)
class TrainingMetricPoint:
    run_id: str
    method_id: str
    policy_seed: int
    timestep: int
    metric: str
    value: float

def extract_training_metrics(
    run_dirs: Sequence[Path],
    *,
    smoke: bool,
) -> tuple[TrainingMetricPoint, ...]: ...
```

- [ ] **Step 1: Write failing matched-comparison tests**

Assert comparison keys include track, scenario cell, episode index, and test seed. Reject missing methods, duplicate rows, profile/contract mismatch, B0 pseudo-seed duplication, differing physical seeds, and mixed Shield modes within a track. Verify physical-episode count, policy-replicate count, means, and sample standard deviation across independent policy-seed means against hand calculations. Assert the policy-replicate standard-deviation cell is empty for B0 and for one-seed smoke PPO output.

- [ ] **Step 2: Write failing TensorBoard extraction tests**

Create tiny event fixtures through `SummaryWriter` for episode reward, each reward component, `train/entropy_loss`, value loss, and explained variance. Assert stable tag mapping, sorted timesteps, finite values, method/seed provenance, conversion to `policy_entropy = -entropy_loss` while retaining the original tag/value, and clear rejection when a required tag is absent in formal mode. In smoke mode, assert unavailable tags are explicitly recorded instead of fabricated. Confirm no training process or simulator is started.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_compare.py tests/unit/evaluation/test_training_metrics.py -q`

Expected: FAIL because comparison and extraction modules do not exist.

- [ ] **Step 4: Implement strict matched aggregation and offline extraction**

Write `metrics/eval_metrics.csv`, `metrics/comparison.csv`, and `metrics/train_metrics.csv` with fixed column order. Keep raw per-episode values in `eval_metrics.csv`; put grouped means/sample dispersion/counts in `comparison.csv`. B0 rows aggregate across episodes only and retain an empty policy-seed cell. Every smoke CSV/report/figure is visibly labeled `SMOKE - NOT A RESEARCH RESULT`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/evaluation/test_compare.py tests/unit/evaluation/test_training_metrics.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the offline-data slice**

Run:

```powershell
git add src/mad_driving/evaluation tests/unit/evaluation
git commit -m "feat: compare matched runs and extract training metrics"
```

---

### Task 8: Generate plots, episode GIFs, and the Markdown report offline

**Files:**
- Create `src/mad_driving/visualization/__init__.py`, `plots.py`, `overlay.py`, and `report.py`.
- Create `tests/unit/visualization/test_plots.py`, `test_overlay.py`, and `test_report.py`.

**Interfaces:**

```python
def write_learning_curve(train_metrics_csv: Path, output_png: Path) -> None: ...
def write_safety_efficiency_plots(eval_metrics_csv: Path, output_dir: Path) -> tuple[Path, ...]: ...
def write_episode_gif(step_jsonl: Path, frames_dir: Path, output_gif: Path) -> None: ...
def write_markdown_report(bundle_dir: Path, output_md: Path) -> None: ...
```

- [ ] **Step 1: Write failing deterministic plot/report tests**

Use fixture CSV/JSONL/lossless RGB frame files. Assert expected PNG/GIF files exist, have non-zero dimensions, and contain deterministic labels/order. Verify the GIF overlay contains method, track, scenario, seed, step, requested/required/executed action, speed, oracle TTC, speed limit, target speed, Shield status, each present Agent's severity/recommended speed, and Critic conflict score. Check the report links the artifact manifest, plan, selected checkpoints, safety/efficiency/comfort/multi-agent/latency tables, plots, and representative episodes. Reject records outside the bundle root or frame path traversal.

- [ ] **Step 2: Write failing import-isolation test**

Run the compare/render module imports in a subprocess that blocks `metadrive` imports. Expected behavior: all offline modules import and operate successfully.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/visualization -q`

Expected: FAIL because visualization modules do not exist.

- [ ] **Step 4: Implement headless deterministic outputs**

Force matplotlib's `Agg` backend before importing pyplot. Use a fixed method order/color map and stable figure size/DPI. Build GIF frames from persisted image paths and text overlays only; never rerun policy decisions. The report must state that smoke results are pipeline validation, not formal scientific conclusions.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/visualization -q`

Expected: PASS.

- [ ] **Step 6: Commit the visualization slice**

Run:

```powershell
git add src/mad_driving/visualization tests/unit/visualization
git commit -m "feat: render phase 6 evaluation artifacts"
```

---

### Task 9: Expose evaluate, compare, and render CLI commands

**Files:**
- Create `src/mad_driving/cli/evaluate.py`, `compare.py`, and `render_episode.py`.
- Create `configs/evaluation/phase6_smoke.yaml`.
- Create `tests/unit/cli/test_evaluate.py`, `test_compare.py`, and `test_render_episode.py`.

**Command contracts:**

```powershell
.venv\Scripts\python.exe -m mad_driving.cli.evaluate --plan configs/evaluation/phase6_smoke.yaml --output evaluations/phase6_smoke --smoke
.venv\Scripts\python.exe -m mad_driving.cli.compare --evaluation evaluations/phase6_smoke
.venv\Scripts\python.exe -m mad_driving.cli.render_episode --evaluation evaluations/phase6_smoke --episode-key proposed_system_42_level1_lead_brake_20000
```

- [ ] **Step 1: Write failing CLI contract tests**

Assert help output, required paths, ordered method overlays, absent destination requirement, concise traceback-free operational errors, JSON success output, and non-zero exit codes for malformed plan/checkpoint/bundle data. Patch orchestration functions; do not launch MetaDrive in unit tests.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/cli/test_evaluate.py tests/unit/cli/test_compare.py tests/unit/cli/test_render_episode.py -q`

Expected: FAIL because the commands and evaluation smoke config do not exist.

- [ ] **Step 3: Implement thin CLI adapters and smoke config**

Keep parsing/path validation in CLI modules and all scientific behavior in evaluation modules. The smoke config must run one deterministic test episode per required method/scenario cell with short horizon and explicit selected smoke checkpoints. If a PPO checkpoint is unavailable, fail clearly; never silently replace PPO with a random or rule policy.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/cli/test_evaluate.py tests/unit/cli/test_compare.py tests/unit/cli/test_render_episode.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the CLI slice**

Run:

```powershell
git add src/mad_driving/cli configs/evaluation tests/unit/cli
git commit -m "feat: add phase 6 evaluation commands"
```

---

### Task 10: Prove the artifact pipeline end to end

**Files:**
- Create `tests/integration/test_phase6_pipeline.py`.
- Create or extend deterministic fake fixtures under `tests/integration/fakes/` if needed.
- Modify Phase 6 modules only when the integration test exposes a missing contract.

- [ ] **Step 1: Write a failing fake end-to-end integration test**

Run a small plan with fake environments and fake checkpoints through plan creation, episode execution, reduction, selection, comparison, plot/GIF/report generation, manifest hashing, and atomic publication. Assert this exact minimum bundle:

```text
config_resolved.yaml
evaluation_plan.yaml
evaluation_manifest.json
model_selection.csv
selected_checkpoints.json
episodes/proposed/system/42/level1_lead_brake/episode_20000_trace.jsonl
episodes/proposed/system/42/level1_lead_brake/episode_20000_summary.json
metrics/train_metrics.csv
metrics/eval_metrics.csv
metrics/comparison.csv
plots/learning_curve.png
plots/collision_rate.png
plots/success_route_completion.png
plots/unnecessary_braking.png
plots/comfort.png
plots/agent_disagreement.png
renders/proposed_42_level1_lead_brake_20000.gif
comparison_report.md
```

Assert the manifest records relative path, byte size, and SHA-256 for every pre-manifest artifact, including source TensorBoard event files. Corrupt one file and prove manifest verification fails. Inject a mid-run exception and prove the final destination remains absent.

- [ ] **Step 2: Run the integration test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_phase6_pipeline.py -q`

Expected: FAIL until the orchestration path connects every Phase 6 component.

- [ ] **Step 3: Implement the minimum orchestration glue**

Add one high-level function such as:

```python
def run_evaluation_bundle(
    *,
    app_config: AppConfig,
    evaluation_config: EvaluationConfig,
    destination: Path,
    environment_factory: EvaluationEnvironmentFactory,
    policy_factory: EvaluationPolicyFactory,
) -> Path: ...
```

It must publish only after schema checks, matched-comparison checks, report generation, and manifest verification pass.

- [ ] **Step 4: Run the integration test twice and verify GREEN/repeatability**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_phase6_pipeline.py -q`

Then run it again with a second absent temp destination. Expected: PASS and identical hashes for all scientific text/CSV/JSON artifacts; image metadata differences, if any, must be eliminated or explicitly normalized.

- [ ] **Step 5: Commit the end-to-end slice**

Run:

```powershell
git add src/mad_driving/evaluation tests/integration/test_phase6_pipeline.py tests/integration/fakes
git commit -m "test: prove phase 6 artifact pipeline end to end"
```

---

### Task 11: Validate with real MetaDrive and a PPO smoke checkpoint

**Files:**
- Create `tests/integration/test_phase6_metadrive_headless.py`.
- Modify evaluation/environment adapters only as required by real APIs.
- Do not commit generated checkpoints, frames, or run bundles.

- [ ] **Step 1: Write a failing real MetaDrive smoke test**

Mark it `integration`. Run one short deterministic episode for B0 and one for proposed, covering at least Lead Brake and Occluded Crossing across the two runs. Assert:

- MetaDrive resets with the planned test scenario and seeds;
- B0 receives visible observation only;
- PPO receives shape `(24,)`;
- step/episode records are strict and finite;
- frame capture produces readable RGB images;
- simulator resources close;
- repeated B0 run has identical scientific trace/actions/outcomes.

- [ ] **Step 2: Run the real smoke and verify RED**

Run: `$env:PYTHONUTF8='1'; .venv\Scripts\python.exe -m pytest tests/integration/test_phase6_metadrive_headless.py -q -s`

Expected: FAIL at the first real adapter mismatch, not with a skipped/soft assertion.

- [ ] **Step 3: Fix only documented MetaDrive adapter mismatches**

Inspect the installed MetaDrive 0.4.3 source before changing behavior. If frame capture or reset options differ from the design, document the API mismatch and smallest compatible adapter in the design doc, then test that adapter. Do not change metric definitions or observation semantics.

- [ ] **Step 4: Produce or locate a valid tiny proposed PPO checkpoint**

Use the existing training CLI in smoke mode and a fresh temporary run directory. Validate its v7 metadata, selected method profile, observation/action schema, and SHA-256 before evaluation. Keep it as a test-generated temporary artifact.

- [ ] **Step 5: Run real MetaDrive smoke and verify GREEN**

Run: `$env:PYTHONUTF8='1'; .venv\Scripts\python.exe -m pytest tests/integration/test_phase6_metadrive_headless.py -q -s`

Expected: PASS.

- [ ] **Step 6: Commit the real integration slice**

Run:

```powershell
git add tests/integration/test_phase6_metadrive_headless.py src/mad_driving docs/superpowers/specs/2026-07-23-phase6-evaluation-design.md
git commit -m "test: validate phase 6 with headless metadrive"
```

---

### Task 12: Document Phase 6 and run repository-wide verification

**Files:**
- Modify `README.md`.
- Modify `docs/multi_agent_driving_mvp_spec.md` only for status/command/artifact clarifications consistent with the approved design.
- Modify any test snapshots caused by intentional documentation/version changes.

- [ ] **Step 1: Document the workflow in simple operational language**

Include method table, decision/system/ablation tracks, fixed scenario matrix, seed split, checkpoint-selection rule, smoke commands, artifact tree, offline rerender command, formal-run deferral, and the statement that the 24-dimensional observation is unchanged.

- [ ] **Step 2: Run placeholder and contract scans**

Run:

```powershell
rg -n "TODO|FIXME|NotImplementedError|pass\s*(#.*)?$|placeholder|random policy" src tests configs docs
rg -n "RESEARCH_CONTRACT_VERSION|OBSERVATION_SCHEMA_VERSION|ACTION_SCHEMA_VERSION" src tests docs
rg -n "b0_rule|b1_nominal|b2_multi_no_review|proposed_no_critic|proposed_no_shield|proposed_no_hazard" src tests configs docs
```

Expected: no accidental placeholders; contract references consistently show v7 and observation/action v1; all methods appear in registry, configs, tests, and docs.

- [ ] **Step 3: Run formatting, lint, and types**

Run:

```powershell
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
```

Expected: PASS.

- [ ] **Step 4: Run the complete suite with branch coverage**

Run: `.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-branch --cov-report=term-missing --cov-fail-under=90`

Expected: all tests PASS and branch coverage is at least 90%.

- [ ] **Step 5: Run the user-facing Phase 6 smoke commands**

Use a fresh output directory and real smoke checkpoint. Run evaluate, compare, and render exactly as documented. Inspect `comparison_report.md`, all CSV/JSON files, every PNG, and one GIF. Confirm the command does not overwrite an existing bundle.

- [ ] **Step 6: Commit documentation and verification fixes**

Run:

```powershell
git add README.md docs src tests configs
git commit -m "docs: document phase 6 evaluation workflow"
```

---

### Task 13: Update `feat-architecture-map-flow-cards`

**Workspace:** `C:\Users\楫屋寿弥\Documents\Codex\2026-07-22\kazu02210679-drive-git\work\drive`

**Files:**
- Modify `architecture-map.json`.
- Regenerate or modify `architecture-map.html` through the repository's established workflow.
- Update architecture-map QA artifacts only if that branch already tracks them and its validator requires them.

- [ ] **Step 1: Verify the separate workspace and preserve unrelated changes**

Run `git status --short --branch` in the architecture-map workspace. Confirm the active branch is `feat-architecture-map-flow-cards`. If unrelated user changes overlap the two map files, stop and report the conflict instead of overwriting them.

- [ ] **Step 2: Add Phase 6 flow cards and edges**

Represent these implemented flows with exact final module/artifact names:

```text
Method Profile -> AgentSuite/Critic/Shield
Evaluation Plan -> Policy Adapter -> MultiAgentSpeedEnv
DecisionTrace + Oracle Truth -> Step JSONL -> Episode Metrics
Validation Matrix -> Checkpoint Selection
Matched Test Episodes -> Comparison CSV
TensorBoard Events -> Training Metrics CSV
Persisted Records/Frames -> Plots + GIFs + Markdown Report
Staging Bundle -> Validation + SHA-256 Manifest -> Atomic Publication
```

Keep the existing flow-card visual language and do not invent components not present in the Phase 6 branch.

- [ ] **Step 3: Run architecture-map validation**

Run: `.venv\Scripts\python.exe tools/validate_architecture_map.py` if that workspace has the same virtual environment; otherwise use its documented Python command.

Expected: PASS with no broken node IDs, dangling edges, schema errors, or stale module paths.

- [ ] **Step 4: Visually inspect the generated HTML**

Open or screenshot `architecture-map.html` using the existing QA workflow. Check card text, clipping, overlap, edge routing, and readability at the branch's standard viewport.

- [ ] **Step 5: Commit and push the architecture-map branch**

Run:

```powershell
git add architecture-map.json architecture-map.html
git commit -m "docs: add phase 6 evaluation flow cards"
git push
```

Record the commit SHA for the Phase 6 handoff.

---

### Task 14: Final review, push, and Pull Request update

- [ ] **Step 1: Review the Phase 6 diff against the approved design**

Run `git diff a95c7db...HEAD --stat` and `git diff a95c7db...HEAD`. Check every design section has code, tests, or an explicit deferred note. Confirm no 24-dimensional observation change and no formal benchmark claims.

- [ ] **Step 2: Use the verification-before-completion skill**

Run fresh final Ruff, format, mypy, complete coverage, fake pipeline, and real MetaDrive smoke commands. Do not rely on earlier outputs.

- [ ] **Step 3: Push the Phase 6 branch**

Run: `git push -u origin feat/phase6-evaluation`

- [ ] **Step 4: Create or update the Draft Pull Request**

The PR body must summarize the three tracks, seven profiles, fixed validation/test protocol, strict artifacts, real MetaDrive evidence, deferred formal run, and architecture-map follow-up commit. Include exact test counts and coverage from the final run.

- [ ] **Step 5: Check CI and address only Phase 6 failures**

Run `gh pr checks` and inspect failed logs. Apply TDD fixes, rerun the affected local gate, push, and wait until Ruff, format, mypy, and pytest coverage are green.

- [ ] **Step 6: Hand off concise results**

Report the Phase 6 branch/PR URL, final commit, architecture-map commit, tests/coverage, real MetaDrive smoke result, generated smoke-bundle location, and the remaining formal five-policy-seed benchmark as the next step.

## Plan Self-Review Checklist

- [ ] Every one of the seven methods is represented by one central profile and one strict overlay.
- [ ] Decision, system, and ablation tracks use the approved method/Shield combinations.
- [ ] Validation and test scenario matrices are fixed and test data never influences checkpoint selection.
- [ ] Reward and safety metrics use method-independent oracle truth.
- [ ] B0 sees only current visible observation and is not duplicated across policy seeds.
- [ ] The stopping-margin formula and latency boundaries are explicit.
- [ ] Strict schemas contain enough provenance to reproduce every episode.
- [ ] Checkpoint tie-break, p95, aggregation, and zero-denominator behavior are deterministic.
- [ ] Partial runs cannot appear as complete published bundles.
- [ ] Offline compare/render/report code runs without MetaDrive.
- [ ] Fake and real integration tests cover the complete path.
- [ ] Observation/action schemas remain v1 while the research contract becomes v7.
- [ ] The formal five-policy-seed run remains deferred and is not misrepresented by smoke results.
- [ ] Architecture-map updates happen after final Phase 6 names/flows are known.
- [ ] No placeholders, undocumented API substitutions, or unrelated features are introduced.
