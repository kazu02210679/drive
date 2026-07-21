# Phase 4.1 Research-Validity Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden Phase 4 episode generation, simulator-state boundaries, reward semantics, failure handling, and experiment artifacts so Phase 5 scenarios can produce valid and reproducible research comparisons without changing the 24-dimensional Coordinator observation shape.

**Architecture:** Introduce explicit role-scoped seed allocation and a no-op-compatible `ScenarioRuntime`, then split each simulator step into agent-visible `SceneObservation` and reward-only `PrivilegedWorldState`. Migrate specialists to isolated multi-claim results, aggregate claims conservatively into the existing 24 slots, and make environment/training faults fail fast. Training factories own train/validation roles, while final test construction remains available only to Phase 6 evaluation code.

**Tech Stack:** Python 3.11, dataclasses, Pydantic v2, NumPy, Gymnasium, MetaDrive, Stable-Baselines3 PPO, pytest, Ruff, mypy, uv.

## Implementation status

Tasks 1-10 are complete at commits `41a584f` through `c668f6f`; their exact RED/GREEN and measured evidence is preserved in `.superpowers/sdd/task-1-report.md` through `task-10-report.md`. Task 11 completed the active-document migration and review remediation. The final parent-held identity follow-up gates report `678 passed`, `90.33%` coverage, `15 passed` real MetaDrive checks, and two new 5,000-requested/6,144-actual PPO smokes whose persisted seed artifacts match exactly. Each writer identity is transferred to parent memory while its descriptor remains open; path occupants and artifact headers are not trust anchors.

The parent retains the final whole-branch review and PR #4 push. Task 11 Step 9 is therefore intentionally not run in this task and is not a completion prerequisite here. Phase 5 actors/Curriculum, Phase 6 work, and the explicit `ttc_valid`, `claim_valid`, `agent_failed`, and `target_actor_present` Observation features remain unimplemented.

## Global Constraints

- Keep the Coordinator observation exactly `(24,)`, `numpy.float32`, finite, and bounded to `[-1, 1]`.
- Keep the four-action order `KEEP=0 < SLOW=1 < PREPARE_STOP=2 < STOP=3`.
- Preserve deterministic Safety Shield monotonicity and emergency STOP behavior.
- Agent, Critic, Coordinator, and Shield code may consume only `SceneObservation`.
- Privileged actor state and outcome labels may be consumed only by reward, evaluation, and debug logging.
- Use standard SB3 PPO rollout-boundary updates; do not add a custom episode-boundary trainer.
- New behavior must follow red-green-refactor TDD. Run each listed RED command before editing production code.
- Do not implement Phase 5 scenario actors, curriculum progression, Phase 6 reports, or observation-schema expansion.
- Keep PR #4 in Draft until all verification in Task 11 passes.

---

### Task 1: Explicit simulation timing and disjoint scenario-role configuration

**Files:**
- Modify: `src/mad_driving/config/models.py`
- Modify: `src/mad_driving/world_model/validation.py`
- Modify: `configs/base.yaml`
- Modify: `configs/train.yaml`
- Modify: `tests/unit/config/test_rl_config.py`
- Modify: `tests/unit/world_model/test_snapshot_builder.py`

**Interfaces:**
- Produces: `SeedRangeConfig`, `ScenarioSplitsConfig`, `MetaDriveConfig.physics_dt_s`, `MetaDriveConfig.decision_repeat`, `MetaDriveConfig.decision_dt_s`.
- Produces: `MetaDriveConfig.lane_width_m=3.5`, mapped to MetaDrive's
  `map_config.lane_width`.
- Produces: `AppConfig.scenarios: ScenarioSplitsConfig`.
- Produces: `decision_interval_s(config) -> float` that validates runtime values against explicit configuration.

- [ ] **Step 1: Write failing timing and split tests**

Add tests equivalent to:

```python
def test_simulation_timing_and_seed_split_defaults() -> None:
    config = AppConfig.model_validate(minimum_app_config())
    assert config.metadrive.physics_dt_s == 0.02
    assert config.metadrive.decision_repeat == 5
    assert config.metadrive.decision_dt_s == 0.10
    assert config.metadrive.lane_width_m == 3.5
    assert config.scenarios.train.range == range(0, 10_000)
    assert config.scenarios.validation.range == range(10_000, 11_000)
    assert config.scenarios.test.range == range(20_000, 21_000)


def test_decision_dt_must_equal_physics_dt_times_repeat() -> None:
    payload = minimum_app_config()
    payload["metadrive"] = {
        "physics_dt_s": 0.02,
        "decision_repeat": 5,
        "decision_dt_s": 0.2,
    }
    with pytest.raises(ValidationError, match="decision_dt_s"):
        AppConfig.model_validate(payload)


def test_scenario_seed_ranges_must_not_overlap() -> None:
    payload = minimum_app_config()
    payload["scenarios"] = {
        "train": {"seed_start": 0, "seed_count": 100},
        "validation": {"seed_start": 50, "seed_count": 10},
        "test": {"seed_start": 200, "seed_count": 10},
    }
    with pytest.raises(ValidationError, match="overlap"):
        AppConfig.model_validate(payload)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/config/test_rl_config.py tests/unit/world_model/test_snapshot_builder.py -q
```

Expected: failures for missing timing and scenario split fields.

- [ ] **Step 3: Implement strict models and MetaDrive mapping**

Add these public shapes, with Pydantic validators for finite positive timing, exact timing consistency via `math.isclose`, positive counts, non-negative starts, and pairwise-disjoint ranges:

```python
class SeedRangeConfig(StrictTypedFrozenModel):
    seed_start: int = Field(ge=0)
    seed_count: PositiveInt

    @property
    def range(self) -> range:
        return range(self.seed_start, self.seed_start + self.seed_count)


class ScenarioSplitsConfig(StrictTypedFrozenModel):
    train: SeedRangeConfig = SeedRangeConfig(seed_start=0, seed_count=10_000)
    validation: SeedRangeConfig = SeedRangeConfig(seed_start=10_000, seed_count=1_000)
    test: SeedRangeConfig = SeedRangeConfig(seed_start=20_000, seed_count=1_000)


class MetaDriveConfig(StrictFrozenModel):
    physics_dt_s: FiniteFloat = Field(default=0.02, gt=0.0)
    decision_repeat: PositiveInt = 5
    decision_dt_s: FiniteFloat = Field(default=0.10, gt=0.0)
    lane_width_m: FiniteFloat = Field(default=3.5, gt=0.0)
```

`AppConfig.metadrive_dict()` must emit MetaDrive keys `physics_world_step_size`,
`decision_repeat`, and `map_config.lane_width`, not the project-only keys
`physics_dt_s`, `decision_dt_s`, or `lane_width_m`.

- [ ] **Step 4: Update both YAML files with exact defaults and remove implicit MetaDrive timing**

Add `physics_dt_s`, `decision_repeat`, `decision_dt_s`, `lane_width_m`, and the three
`scenarios` ranges from the design. Keep `start_seed` and `num_scenarios` only for
non-training smoke compatibility; role-aware factories override them in Task 9.

- [ ] **Step 5: Run focused tests and static checks**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/config tests/unit/world_model/test_snapshot_builder.py -q
.venv\Scripts\ruff.exe check src/mad_driving/config src/mad_driving/world_model tests/unit/config tests/unit/world_model
.venv\Scripts\mypy.exe src/mad_driving/config src/mad_driving/world_model
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add src/mad_driving/config/models.py src/mad_driving/world_model/validation.py configs/base.yaml configs/train.yaml tests/unit/config/test_rl_config.py tests/unit/world_model/test_snapshot_builder.py
git commit -m "feat: define simulation timing and seed splits"
```

---

### Task 2: Reproducible role-scoped seed identities

**Files:**
- Create: `src/mad_driving/scenarios/__init__.py`
- Create: `src/mad_driving/scenarios/seeding.py`
- Create: `tests/unit/scenarios/test_seeding.py`

**Interfaces:**
- Consumes: `SeedRangeConfig` from Task 1.
- Produces: `EnvironmentRole = Literal["train", "validation", "test"]`.
- Produces: immutable `EpisodeSeeds` and `EpisodeSeedAllocator`.

- [ ] **Step 1: Write failing deterministic allocation tests**

```python
def test_allocator_is_reproducible_and_role_bounded() -> None:
    split = SeedRangeConfig(seed_start=10_000, seed_count=1_000)
    first = EpisodeSeedAllocator("validation", split, worker_index=2).allocate(42)
    second = EpisodeSeedAllocator("validation", split, worker_index=2).allocate(42)
    assert first == second
    assert first.episode_rng_seed == 42
    assert 10_000 <= first.metadrive_scenario_index < 11_000
    assert 10_000 <= first.scenario_parameter_seed < 11_000


def test_role_or_worker_changes_derived_seed_identity() -> None:
    split = SeedRangeConfig(seed_start=0, seed_count=10_000)
    assert (
        EpisodeSeedAllocator("train", split, worker_index=0).allocate(42)
        != EpisodeSeedAllocator("train", split, worker_index=1).allocate(42)
    )
```

- [ ] **Step 2: Run test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_seeding.py -q`

Expected: import failure because `mad_driving.scenarios.seeding` does not exist.

- [ ] **Step 3: Implement exact SeedSequence derivation**

```python
@dataclass(frozen=True)
class EpisodeSeeds:
    episode_rng_seed: int
    metadrive_scenario_index: int
    scenario_parameter_seed: int


_ROLE_CODES: dict[EnvironmentRole, int] = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


@dataclass(frozen=True)
class EpisodeSeedAllocator:
    role: EnvironmentRole
    seed_range: SeedRangeConfig
    worker_index: int

    def allocate(self, episode_rng_seed: int) -> EpisodeSeeds:
        sequence = np.random.SeedSequence(
            [episode_rng_seed, _ROLE_CODES[self.role], self.worker_index]
        )
        road, parameters = sequence.spawn(2)
        return EpisodeSeeds(
            episode_rng_seed=episode_rng_seed,
            metadrive_scenario_index=self._bounded(road),
            scenario_parameter_seed=self._bounded(parameters),
        )
```

Generate each unsigned integer with `child.generate_state(1, dtype=np.uint32)[0]`, reduce modulo `seed_count`, and offset by `seed_start`. Validate non-negative episode seed and worker index.

- [ ] **Step 4: Run tests and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_seeding.py -q
.venv\Scripts\ruff.exe check src/mad_driving/scenarios tests/unit/scenarios
.venv\Scripts\mypy.exe src/mad_driving/scenarios
git add src/mad_driving/scenarios tests/unit/scenarios/test_seeding.py
git commit -m "feat: add role scoped episode seeds"
```

Expected: all pass and one focused commit is created.

---

### Task 3: ScenarioRuntime lifecycle boundary

**Files:**
- Create: `src/mad_driving/interfaces/scene_frame.py`
- Modify: `src/mad_driving/interfaces/__init__.py`
- Create: `src/mad_driving/scenarios/runtime.py`
- Modify: `src/mad_driving/scenarios/__init__.py`
- Create: `tests/unit/scenarios/test_runtime.py`
- Modify: `tests/unit/interfaces/test_models.py`

**Interfaces:**
- Consumes: `EpisodeSeeds`, the existing `DrivingEnvironment` protocol, and raw MetaDrive info mappings.
- Produces: validated `OcclusionRegion` and `RoadContext` primitives before scenario code consumes them.
- Produces: `ScenarioRuntime`, `ScenarioState`, `ScenarioStepResult`, `ScenarioObservationContext`, and `NoOpScenarioRuntime`.

- [ ] **Step 1: Write failing no-op and context validation tests**

```python
class FakeEnvironment:
    pass


def test_noop_runtime_has_stable_lifecycle_outputs() -> None:
    runtime = NoOpScenarioRuntime("phase4_noop")
    seeds = EpisodeSeeds(
        episode_rng_seed=42,
        metadrive_scenario_index=7,
        scenario_parameter_seed=11,
    )
    state = runtime.reset(FakeEnvironment(), seeds=seeds)
    runtime.after_simulator_reset(FakeEnvironment(), state)
    runtime.before_step(FakeEnvironment(), state, step_index=1)
    result = runtime.after_step(
        FakeEnvironment(), state, step_index=1, raw_info={}
    )
    assert result == ScenarioStepResult(success=False, failure=False)
    assert runtime.observation_context(state) == ScenarioObservationContext(
        scenario_id="phase4_noop",
        stop_required=False,
        occlusion_regions=(),
        distance_to_conflict_point_m=None,
        intersection_entry_prohibited=False,
        visible_actor_ids=None,
    )


def test_active_occlusion_requires_visibility_metadata() -> None:
    region = OcclusionRegion(
        region_id="building-corner",
        boundary_points_xy_m=((0.0, 0.0), (1.0, 0.0)),
    )
    with pytest.raises(ValueError, match="visible_actor_ids"):
        ScenarioObservationContext(
            scenario_id="occluded",
            occlusion_regions=(region,),
            visible_actor_ids=None,
        )
```

- [ ] **Step 2: Run test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_runtime.py -q`

Expected: import failure for the new runtime types.

- [ ] **Step 3: Implement immutable lifecycle types and protocol**

First define frozen `OcclusionRegion(region_id, boundary_points_xy_m)` and
`RoadContext(stop_required, distance_to_conflict_point_m,
intersection_entry_prohibited)` dataclasses in `interfaces/scene_frame.py`. Validate unique
non-empty region IDs, at least two finite boundary points, and finite optional conflict
distance. Then define the runtime types.

`ScenarioState` contains `scenario_id`, `EpisodeSeeds`, and an immutable parameter mapping.
Freeze mappings with `MappingProxyType(dict(parameters))`. `ScenarioObservationContext`
converts actor IDs to `frozenset[str]` and enforces the fail-closed occlusion rule. The
no-op runtime must have no mutable episode state and no simulator side effects.

- [ ] **Step 4: Run tests and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/scenarios tests/unit/interfaces/test_models.py -q
.venv\Scripts\ruff.exe check src/mad_driving/scenarios tests/unit/scenarios
.venv\Scripts\mypy.exe src/mad_driving/scenarios src/mad_driving/interfaces/scene_frame.py
git add src/mad_driving/scenarios src/mad_driving/interfaces/scene_frame.py src/mad_driving/interfaces/__init__.py tests/unit/scenarios tests/unit/interfaces/test_models.py
git commit -m "feat: add scenario runtime lifecycle"
```

---

### Task 4: Split agent observation from privileged simulator truth

**Files:**
- Modify: `src/mad_driving/interfaces/scene_frame.py`
- Modify: `src/mad_driving/interfaces/scene_snapshot.py`
- Modify: `src/mad_driving/interfaces/__init__.py`
- Modify: `src/mad_driving/interfaces/defensive_validation.py`
- Modify: `src/mad_driving/world_model/snapshot_builder.py`
- Modify: `src/mad_driving/world_model/__init__.py`
- Modify: `tests/unit/interfaces/test_models.py`
- Modify: `tests/unit/world_model/test_snapshot_builder.py`
- Modify: `tests/unit/agents/factories.py`

**Interfaces:**
- Consumes: `EpisodeSeeds` and `ScenarioObservationContext`.
- Consumes: `RoadContext` and `OcclusionRegion` from Task 3.
- Produces: `SceneObservation`, `PrivilegedWorldState`, `SceneFrame`.
- Produces: `SceneSnapshot = SceneObservation` as a temporary import alias only; new production annotations use `SceneObservation`.
- Produces: `SceneSnapshotBuilder.build` returning `SceneFrame`.

- [ ] **Step 1: Write failing visibility and privileged-label tests**

```python
def test_hidden_actor_exists_only_in_privileged_state() -> None:
    env = make_env()
    frame = build_frame(
        env,
        visible_actor_ids=frozenset({"z-vehicle"}),
    )
    assert tuple(a.actor_id for a in frame.observation.visible_actors) == ("z-vehicle",)
    assert tuple(a.actor_id for a in frame.privileged.all_actors) == (
        "a-vehicle",
        "z-vehicle",
    )
    assert not hasattr(frame.observation, "collision_occurred")


def test_privileged_outcomes_combine_raw_info_and_vehicle_state() -> None:
    env = make_env()
    frame = build_frame(env, raw_info={"out_of_road": True, "arrive_dest": True})
    assert frame.privileged.off_road is True
    assert frame.privileged.arrived is True
```

Also add model tests rejecting hidden kinematics in `visible_actors`, non-finite occlusion geometry, invalid heading, and mutable seed/context inputs.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/interfaces/test_models.py tests/unit/world_model/test_snapshot_builder.py -q
```

Expected: missing frame types and old builder return shape failures.

- [ ] **Step 3: Implement frame types and builder separation**

`SceneObservation` keeps the agent-visible fields currently used by specialists, Coordinator, Shield, and control. Rename `actors` to `visible_actors`, `previous_action` to `previous_executed_action`, and move collision/arrival/off-road/scenario labels to `PrivilegedWorldState`.

Builder behavior filters a fully constructed `all_actors` tuple before creating the
agent-visible observation:

```python
visible_actors = tuple(
    actor
    for actor in all_actors
    if context.visible_actor_ids is None
    or actor.actor_id in context.visible_actor_ids
)
privileged = PrivilegedWorldState(
    all_actors=all_actors,
    collision_occurred=self._collision_occurred(ego_vehicle),
    collision_kind=self._collision_kind(raw_info, ego_vehicle),
    off_road=bool(raw_info.get("out_of_road", False)) or self._off_road(ego_vehicle),
    arrived=bool(raw_info.get("arrive_dest", False)),
    scenario_success=scenario_result.success,
    scenario_failure=scenario_result.failure,
)
return SceneFrame(observation=observation, privileged=privileged)
```

Normalize heading to `[-pi, pi)`. Set actor relative lateral positive to ego-left. Determine `same_lane` from canonical lane index plus lateral position inside lane width, using lane width when exposed and the configured fallback otherwise.

- [ ] **Step 4: Migrate test factories to construct `SceneObservation` and `SceneFrame`**

Keep `make_snapshot()` as a test-only compatibility helper returning `SceneObservation` so existing focused agent tests can be migrated incrementally. Add `make_frame()` for reward and environment tests.

- [ ] **Step 5: Run tests and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/interfaces tests/unit/world_model -q
.venv\Scripts\ruff.exe check src/mad_driving/interfaces src/mad_driving/world_model tests/unit/interfaces tests/unit/world_model
.venv\Scripts\mypy.exe src/mad_driving/interfaces src/mad_driving/world_model
git add src/mad_driving/interfaces src/mad_driving/world_model tests/unit/interfaces tests/unit/world_model tests/unit/agents/factories.py
git commit -m "refactor: separate observed and privileged state"
```

---

### Task 5: Isolate specialist failures and retain multiple RiskClaims

**Files:**
- Modify: `src/mad_driving/agents/protocol.py`
- Modify: `src/mad_driving/agents/nominal.py`
- Modify: `src/mad_driving/agents/hazard.py`
- Modify: `src/mad_driving/agents/rule.py`
- Modify: `src/mad_driving/agents/critic.py`
- Modify: `src/mad_driving/agents/suite.py`
- Modify: `src/mad_driving/agents/claim_factory.py`
- Modify: `src/mad_driving/agents/__init__.py`
- Modify: `tests/unit/agents/test_nominal.py`
- Modify: `tests/unit/agents/test_hazard.py`
- Modify: `tests/unit/agents/test_rule.py`
- Modify: `tests/unit/agents/test_critic.py`
- Modify: `tests/unit/agents/test_suite.py`
- Modify: `tests/unit/agents/factories.py`

**Interfaces:**
- Consumes: `SceneObservation` from Task 4.
- Produces: `DrivingAgent.analyze` returning `tuple[RiskClaim, ...]`.
- Produces: immutable `AgentAnalysisResult(claims, failed_agent_ids, errors, review)`.
- Produces: `analyze_safely(suite, observation) -> AgentAnalysisResult`.
- Produces test helper: `make_analysis(claims=(), failed_agent_ids=(), errors=(),
  review=None) -> AgentAnalysisResult`. When review is omitted it derives a finite neutral
  `CriticReview` from the provided claim IDs.

- [ ] **Step 1: Write failing multi-claim and isolated-failure tests**

```python
def test_hazard_returns_up_to_three_deterministically_ordered_claims() -> None:
    actors = tuple(
        make_actor(
            f"lead-{distance}",
            longitudinal_m=float(distance),
            longitudinal_speed_mps=0.0,
        )
        for distance in (15, 5, 10)
    )
    observation = make_snapshot(actors=actors)
    claims = HazardAgent(HazardAgentConfig()).analyze(observation)
    assert 1 <= len(claims) <= 3
    assert claims == HazardAgent(HazardAgentConfig()).analyze(observation)
    assert tuple(claim.target_actor_id for claim in claims) == (
        "lead-5",
        "lead-10",
        "lead-15",
    )


def test_one_agent_failure_preserves_surviving_claims() -> None:
    suite = AgentSuite(
        nominal=FailingAgent("nominal"),
        hazard=RecordingAgent("hazard"),
        rule=RecordingAgent("rule"),
        critic=RecordingCritic(),
    )
    result = analyze_safely(suite, make_snapshot())
    assert result.failed_agent_ids == ("nominal",)
    assert tuple(claim.agent_id for claim in result.claims) == ("hazard", "rule")
    assert "agent_analysis_failed:nominal" in result.review.reasons
```

Add tests that intentional ablation is represented by an absent configured agent and not by `failed_agent_ids`, all returned claim IDs are unique, and Critic receives every surviving claim.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/agents -q`

Expected: protocol/return-shape failures.

- [ ] **Step 3: Implement candidate ordering and per-agent exception boundaries**

Each agent returns a tuple. Neutral scenes still return one neutral claim. Nominal and Hazard retain their current candidate calculations but return the first three after this shared key:

```python
def claim_safety_key(claim: RiskClaim) -> tuple[object, ...]:
    return (
        not claim.hard_stop_required,
        -claim.severity,
        claim.min_ttc_s is None,
        math.inf if claim.min_ttc_s is None else claim.min_ttc_s,
        math.inf if claim.stopping_margin_m is None else claim.stopping_margin_m,
        claim.recommended_max_speed_mps,
        claim.target_actor_id or "",
        claim.event_type,
    )
```

`AgentSuite` catches exceptions around each `agent.analyze` call, sanitizes errors to `"<agent_id>:<ExceptionType>:<message>"`, preserves other claims, invokes Critic once, and augments review reasons with `agent_analysis_failed:<id>`. It does not synthesize a normal claim for a failed agent; Shield missing-agent logic remains effective.

- [ ] **Step 4: Run tests and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/agents -q
.venv\Scripts\ruff.exe check src/mad_driving/agents tests/unit/agents
.venv\Scripts\mypy.exe src/mad_driving/agents
git add src/mad_driving/agents tests/unit/agents
git commit -m "feat: isolate agents and retain multiple claims"
```

---

### Task 6: Conservatively aggregate claims into the unchanged 24 slots

**Files:**
- Modify: `src/mad_driving/coordinator/observation.py`
- Modify: `src/mad_driving/coordinator/rule_based.py`
- Modify: `tests/unit/coordinator/test_observation.py`
- Modify: `tests/unit/coordinator/test_rule_based.py`

**Interfaces:**
- Consumes: multiple claims for each specialist and `AgentAnalysisResult.review`.
- Produces: private immutable `_AggregatedClaim` and `aggregate_agent_claims(agent_id, claims)`.
- Preserves: `ObservationBuilder.build` returning `NDArray[np.float32]` with shape `(24,)`.

- [ ] **Step 1: Write failing field-wise aggregation tests**

```python
def test_multiple_hazard_claims_are_aggregated_field_wise() -> None:
    claims = (
        make_claim("hazard", min_ttc_s=4.0, severity=0.9,
                   stopping_margin_m=8.0, recommended_max_speed_mps=10.0),
        make_claim("hazard", min_ttc_s=1.0, severity=0.4,
                   stopping_margin_m=-2.0, recommended_max_speed_mps=4.0),
    )
    complete_claims = (make_claim("nominal"), *claims, make_claim("rule"))
    review = CriticReview(
        conflict_score=0.0,
        unresolved_conflict=False,
        max_severity=0.9,
        supported_agent_ids=("nominal", "hazard", "rule"),
        challenged_claim_ids=(),
        reasons=(),
    )
    observation = ObservationBuilder(ObservationConfig()).build(
        make_snapshot(), complete_claims, review
    )
    assert observation[10] == pytest.approx(0.1)
    assert observation[11] == pytest.approx(-0.04)
    assert observation[12] == pytest.approx(0.9)
    assert observation[14] == pytest.approx(0.1)


def test_observation_stays_24_float32_with_duplicate_agent_claims() -> None:
    claims = (
        make_claim("nominal", min_ttc_s=4.0),
        make_claim("nominal", min_ttc_s=2.0),
        make_claim("hazard"),
        make_claim("rule"),
    )
    review = CriticReview(
        conflict_score=0.0,
        unresolved_conflict=False,
        max_severity=0.0,
        supported_agent_ids=("nominal", "hazard", "rule"),
        challenged_claim_ids=(),
        reasons=(),
    )
    value = ObservationBuilder(ObservationConfig()).build(
        make_snapshot(), claims, review
    )
    assert value.shape == (24,)
    assert value.dtype == np.float32
    assert np.isfinite(value).all()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/coordinator -q`

Expected: current duplicate-agent rejection fails both tests.

- [ ] **Step 3: Implement conservative aggregation**

For one agent's valid claims compute minimum finite TTC, minimum finite margin, maximum severity/probability, minimum recommended speed, any hard stop, and minimum confidence. Return `None` only when no claim exists for that agent. Remove duplicate-agent rejection while retaining defensive validation of every claim. Rule-based Coordinator considers all individual claims and remains safety-monotonic.

- [ ] **Step 4: Run tests and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/coordinator -q
.venv\Scripts\ruff.exe check src/mad_driving/coordinator tests/unit/coordinator
.venv\Scripts\mypy.exe src/mad_driving/coordinator
git add src/mad_driving/coordinator tests/unit/coordinator
git commit -m "feat: aggregate claims into fixed observation"
```

---

### Task 7: Current-state reward and privileged outcome inputs

**Files:**
- Modify: `src/mad_driving/config/models.py`
- Modify: `src/mad_driving/envs/reward.py`
- Modify: `configs/base.yaml`
- Modify: `configs/train.yaml`
- Modify: `tests/unit/config/test_rl_config.py`
- Modify: `tests/unit/envs/test_reward.py`

**Interfaces:**
- Consumes: previous/next `SceneFrame`, `AgentAnalysisResult`, executed action, and decision dt.
- Produces: `RewardContext(previous_frame, next_frame, analysis, executed_action, shield_intervened, decision_interval_s)`.
- Removes: `RewardConfig.unnecessary_brake_lookahead_steps` and episode-local safe-brake streak.

- [ ] **Step 1: Write failing no-actor, off-road, and no-future reward tests**

```python
def test_no_actor_safe_scene_penalizes_unnecessary_stop_immediately() -> None:
    context = make_context(
        analysis=make_analysis(
            claims=(
                make_claim("nominal", min_ttc_s=None, severity=0.0),
                make_claim("hazard", min_ttc_s=None, severity=0.0),
                make_claim("rule", min_ttc_s=None, severity=0.0),
            ),
            failed_agent_ids=(),
        ),
        executed_action=DrivingAction.STOP,
    )
    result = RewardCalculator(RewardConfig()).calculate(context)
    assert result.components["unnecessary_brake_penalty"] == pytest.approx(-0.6)


def test_raw_out_of_road_privileged_state_drives_penalty() -> None:
    context = make_context(next_frame=make_frame(off_road=True))
    result = RewardCalculator(RewardConfig()).calculate(context)
    assert result.components["offroad_penalty"] == -100.0


def test_failed_hazard_suppresses_unnecessary_brake_judgment() -> None:
    context = make_context(
        analysis=make_analysis(failed_agent_ids=("hazard",)),
        executed_action=DrivingAction.STOP,
    )
    assert RewardCalculator(RewardConfig()).calculate(context).components[
        "unnecessary_brake_penalty"
    ] == 0.0
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/envs/test_reward.py tests/unit/config/test_rl_config.py -q`

Expected: old lookahead and `min_ttc_s is not None` behavior fail.

- [ ] **Step 3: Implement stateless current-state penalty**

Use `safe_ttc = min_ttc_s is None or min_ttc_s >= safe_threshold`. Require Hazard and Rule to be present and not failed. Read collision, arrival, and off-road only from `next_frame.privileged`. Retain arrival's one-time episode guard; remove only `_safe_brake_streak`.

- [ ] **Step 4: Remove YAML field, run tests, and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/envs/test_reward.py tests/unit/config -q
.venv\Scripts\ruff.exe check src/mad_driving/envs/reward.py tests/unit/envs/test_reward.py
.venv\Scripts\mypy.exe src/mad_driving/envs/reward.py
git add src/mad_driving/config/models.py src/mad_driving/envs/reward.py configs/base.yaml configs/train.yaml tests/unit/config/test_rl_config.py tests/unit/envs/test_reward.py
git commit -m "fix: use current privileged reward state"
```

---

### Task 8: Integrate seed progression, ScenarioRuntime, frames, and fail-fast semantics

**Files:**
- Modify: `src/mad_driving/envs/multi_agent_speed_env.py`
- Modify: `src/mad_driving/envs/__init__.py`
- Modify: `tests/unit/envs/test_multi_agent_speed_env.py`
- Modify: `tests/integration/test_rl_metadrive_headless.py`

**Interfaces:**
- Consumes: `EnvironmentRole`, worker index, `EpisodeSeedAllocator`, `ScenarioRuntime`, `SceneFrame`, `AgentAnalysisResult`.
- Produces: `MultiAgentSpeedEnv` constructor parameters `config`, `role`, `worker_index`,
  `scenario_runtime_factory`, and the existing injectable factories.
- Preserves: Gymnasium `reset` and `step` signatures and 24-dimensional spaces.

- [ ] **Step 1: Write failing implicit-reset seed sequence tests**

```python
def test_implicit_resets_advance_reproducible_episode_seed_sequence() -> None:
    first = make_env()
    second = make_env()
    try:
        first_values = [first.env.reset(seed=42)[1]["episode_rng_seed"]]
        second_values = [second.env.reset(seed=42)[1]["episode_rng_seed"]]
        first_values.extend(first.env.reset()[1]["episode_rng_seed"] for _ in range(3))
        second_values.extend(second.env.reset()[1]["episode_rng_seed"] for _ in range(3))
        assert first_values == second_values
        assert len(set(first_values)) == len(first_values)
    finally:
        first.env.close()
        second.env.close()
```

Add a fake VecEnv auto-reset regression test proving the second episode does not reuse 42.

- [ ] **Step 2: Write failing hook-order and actual-seed logging tests**

Assert exact order `runtime.reset`, simulator reset, `after_simulator_reset`, initial frame, then `before_step`, simulator step, `after_step`, next frame. Assert reset and step info contain role, worker index, all three seed identities, and the actual simulator scenario index. Make the fake simulator return a mismatching index and expect an exception.

- [ ] **Step 3: Write failing internal-error propagation tests**

Replace existing tests that expect zero observation plus truncation with:

```python
@pytest.mark.parametrize("failure", ["simulator", "snapshot", "reward", "observation"])
def test_internal_failure_closes_simulator_and_propagates(failure: str) -> None:
    harness = make_env_with_existing_failure_fake(failure)
    harness.env.reset(seed=42)
    with pytest.raises(RuntimeError, match=failure):
        harness.env.step(DrivingAction.KEEP)
    assert harness.simulator.close_calls == 1
```

- [ ] **Step 4: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/envs/test_multi_agent_speed_env.py -q`

Expected: seed repetition, missing runtime hooks, and old truncation behavior fail.

For `make_env_with_existing_failure_fake`, add one test helper that selects the existing
`FakeSimulator(fail_on_step=True)`, failing `RecordingSnapshotBuilder`, failing
`RecordingRewardCalculator`, or failing `RecordingObservationBuilder` according to the
parameter. It returns the existing `EnvHarness`; it does not add production hooks.

- [ ] **Step 5: Implement first-reset and implicit-reset RNG rules**

Track whether the Gymnasium RNG has been initialized. On explicit `seed`, call `super().reset(seed=seed)` and use that value. On first implicit reset, call `super().reset(seed=config.seed)` and use `config.seed`. On later implicit reset, call `super().reset(seed=None)` and draw with:

```python
episode_rng_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
```

Allocate all three seed identities, execute runtime hooks, build frames, analyze specialists, calculate reward, and publish metadata.

- [ ] **Step 6: Remove safe-observation recovery and propagate faults**

Delete `_safe_observation` and `_build_observation_safely`. Wrap owned resource cleanup so cleanup failures are added as notes without masking the primary exception, matching training cleanup behavior. Natural horizon remains `truncated`; scenario success/failure and collision remain `terminated`.

- [ ] **Step 7: Run unit and real MetaDrive integration tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/envs/test_multi_agent_speed_env.py -q
.venv\Scripts\python.exe -m pytest tests/integration/test_rl_metadrive_headless.py -q
.venv\Scripts\ruff.exe check src/mad_driving/envs tests/unit/envs tests/integration/test_rl_metadrive_headless.py
.venv\Scripts\mypy.exe src/mad_driving/envs
```

Expected: all pass and repeated explicit seeds still reproduce identical initial observations/traces.

- [ ] **Step 8: Commit**

```powershell
git add src/mad_driving/envs tests/unit/envs/test_multi_agent_speed_env.py tests/integration/test_rl_metadrive_headless.py
git commit -m "feat: harden environment episode lifecycle"
```

---

### Task 9: Role-aware factories and actual MetaDrive scenario identity

**Files:**
- Modify: `src/mad_driving/envs/control_metadrive_env.py`
- Modify: `src/mad_driving/training/train.py`
- Modify: `tests/unit/training/test_train.py`
- Modify: `tests/integration/test_control_metadrive_headless.py`
- Modify: `tests/integration/test_rl_metadrive_headless.py`

**Interfaces:**
- Changes: `EnvironmentFactory(config, *, role, worker_index)`.
- Produces: role-specific MetaDrive config with `start_seed` and `num_scenarios` from `AppConfig.scenarios`.
- Produces: train workers with `role="train"`; EvalCallback environment with `role="validation"`.

- [ ] **Step 1: Write failing role-factory tests**

```python
def test_training_constructs_train_workers_and_validation_eval(tmp_path: Path) -> None:
    factory = RecordingEnvironmentFactory()
    run_with_fakes(
        make_config(num_envs=3), tmp_path / "role-aware", env_factory=factory
    )
    assert [(call.role, call.worker_index) for call in factory.calls] == [
        ("train", 0),
        ("train", 1),
        ("train", 2),
        ("validation", 0),
    ]


def test_test_role_is_not_constructed_by_training(tmp_path: Path) -> None:
    factory = RecordingEnvironmentFactory()
    run_with_fakes(make_config(), tmp_path / "role-aware", env_factory=factory)
    assert all(call.role != "test" for call in factory.calls)
```

Add integration assertions that validation MetaDrive indices stay in `[10000, 11000)` and reported `metadrive_scenario_index` equals the simulator's returned index.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/training/test_train.py tests/integration/test_rl_metadrive_headless.py -q
```

Expected: factory signature and role assertions fail.

- [ ] **Step 3: Implement role-aware factory ownership**

Pass role and worker index through cloudpickle-safe environment thunks. Build a role-specific copy of the MetaDrive dictionary with the role range. Remove modulo wrapping from `ControlMetaDriveEnv.reset`; reject indices outside the configured MetaDrive range and trust the allocator to provide valid values. Return the actual index in reset info.

- [ ] **Step 4: Add emergency STOP end-to-end assertion**

Extend control integration coverage so a Shield-enforced STOP reaches `LaneKeepingLongitudinalPolicy` and yields the configured emergency deceleration path, while monitor mode leaves the requested action unchanged.

- [ ] **Step 5: Run tests and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/training/test_train.py tests/integration/test_control_metadrive_headless.py tests/integration/test_rl_metadrive_headless.py -q
.venv\Scripts\ruff.exe check src/mad_driving/envs src/mad_driving/training tests/unit/training tests/integration
.venv\Scripts\mypy.exe src/mad_driving/envs src/mad_driving/training
git add src/mad_driving/envs/control_metadrive_env.py src/mad_driving/training/train.py tests/unit/training/test_train.py tests/integration
git commit -m "feat: separate train and validation environments"
```

---

### Task 10: Strict run ownership and resume provenance

**Files:**
- Create: `src/mad_driving/training/metadata.py`
- Modify: `src/mad_driving/training/train.py`
- Modify: `src/mad_driving/cli/train.py`
- Modify: `src/mad_driving/training/__init__.py`
- Modify: `tests/unit/training/test_train.py`
- Modify: `tests/unit/cli/test_train.py`
- Modify: `tests/integration/test_ppo_checkpoint.py`

**Interfaces:**
- Produces: `RunMetadata`, `ResumeMetadata`, `sha256_file(path)`, `validate_resume_contract(model, config, metadata)`.
- Produces: `research_contract_version=2`, `observation_schema_version=1` metadata JSON.
- Changes: new training and resume require an empty destination; resume source may be in a different read-only run.
- Produces test helper: `seed_compatible_source_run(tmp_path, **ppo_overrides)` creates a
  source directory containing a fake checkpoint, resolved config, and version-2 metadata,
  and returns a frozen object with `run_dir` and `checkpoint` paths.

- [ ] **Step 1: Write failing no-write-on-nonempty tests**

```python
def test_nonempty_run_directory_is_rejected_before_any_write(tmp_path: Path) -> None:
    run_dir = tmp_path / "occupied"
    run_dir.mkdir()
    marker = run_dir / "keep.txt"
    marker.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        run_with_fakes(make_config(), run_dir)
    assert marker.read_text(encoding="utf-8") == "original"
    assert list(run_dir.iterdir()) == [marker]
```

Update the old same-run resume tests: the resume checkpoint stays in a source run, while the continuation writes to a distinct empty destination.

- [ ] **Step 2: Write failing provenance and mismatch tests**

```python
def test_resume_records_parent_hash_config_diff_and_start_step(tmp_path: Path) -> None:
    source = seed_compatible_source_run(tmp_path)
    destination = tmp_path / "continued"
    run_with_fakes(make_config(), destination, resume_from=source.checkpoint)
    metadata = json.loads((destination / "run_metadata.json").read_text())
    assert metadata["research_contract_version"] == 2
    assert metadata["resume"]["parent_checkpoint_sha256"] == sha256_file(source.checkpoint)
    assert metadata["resume"]["start_num_timesteps"] == 12_500
    assert metadata["resume"]["config_diff"] == {}


def test_resume_rejects_ppo_hyperparameter_mismatch_before_learning(tmp_path: Path) -> None:
    source = seed_compatible_source_run(tmp_path, learning_rate=0.0003)
    with pytest.raises(ValueError, match="learning_rate"):
        run_with_fakes(
            make_config(learning_rate=0.001),
            tmp_path / "continued",
            resume_from=source.checkpoint,
        )
    assert FakePPO.instances[0].learn_kwargs is None
```

Also test missing/legacy metadata rejection, observation/action space mismatch, and byte-for-byte preservation of the source run.

- [ ] **Step 3: Run tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/training/test_train.py tests/unit/cli/test_train.py tests/integration/test_ppo_checkpoint.py -q
```

Expected: current directory reuse and unrestricted resume behavior fail.

- [ ] **Step 4: Implement metadata and preflight validation**

Preflight before creating directories:

```python
def require_empty_run_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Run directory is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory is non-empty: {path}")
```

Serialize metadata transactionally through a sibling temporary file and `os.replace`. Compare the exact hyperparameter list from the design after `PPO.load` and before `learn`. Record parent hash/path/config/diff/start step and both schema versions. Reject legacy sources without `run_metadata.json` version 2.

- [ ] **Step 5: Run tests and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/training tests/unit/cli/test_train.py tests/integration/test_ppo_checkpoint.py -q
.venv\Scripts\ruff.exe check src/mad_driving/training src/mad_driving/cli/train.py tests/unit/training tests/unit/cli/test_train.py
.venv\Scripts\mypy.exe src/mad_driving/training src/mad_driving/cli/train.py
git add src/mad_driving/training src/mad_driving/cli/train.py tests/unit/training tests/unit/cli/test_train.py tests/integration/test_ppo_checkpoint.py
git commit -m "feat: enforce training artifact provenance"
```

---

### Task 11: Specification migration, full verification, and PR update

**Files:**
- Modify: `docs/multi_agent_driving_mvp_spec.md`
- Modify: `README.md`
- Modify: `docs/phase4_implementation_log.md`
- Modify: `docs/implementation_plan.md`
- Modify: `docs/superpowers/plans/2026-07-21-phase4-research-validity-hardening.md`
- Modify: affected tests if documentation commands expose a genuine mismatch

**Interfaces:**
- Documents: three specialists plus one Critic, explicit timing, standard PPO update semantics, seed identities/splits, ScenarioRuntime, privileged-state boundary, current reward, Shield comparison contracts, coordinates, and fresh-run requirements.
- Records: 24-dimensional validity flags as a deferred versioned observation-schema change.
- Persists: actual initial and VecEnv auto-reset seed identities in role/worker JSONL artifacts, with validated metadata inventory and version-2 resume compatibility.

- [x] **Step 1: Update the top-level specification and README**

Make these exact semantic corrections:

- replace “4 specialist agents” with “3 specialist agents + 1 Critic”;
- remove the prohibition on mid-episode neural-network updates;
- define `physics_dt_s=0.02`, `decision_repeat=5`, and `decision_dt_s=0.10`;
- replace one-claim protocols with one-to-three claims and 24-slot aggregation;
- state that hidden actor kinematics never enter agent-visible structures;
- define train/validation/test ranges and five training seeds;
- define decision comparison as all-monitor and system comparison as all-enforce;
- define current-state unnecessary braking and final-test isolation;
- document world/body/lane coordinate signs;
- document fresh run directory and resume provenance requirements.

- [x] **Step 2: Run documentation consistency scans**

Run:

```powershell
rg -n "4つの専門Agent|エピソード途中のニューラルネットワーク更新|unnecessary_brake_lookahead_steps|1件のRiskClaim|24次元Observation" README.md docs configs src tests
rg -n "train.*validation.*test|physics_dt_s|decision_dt_s|PrivilegedWorldState|ScenarioRuntime|research_contract_version" README.md docs configs src
```

Expected: obsolete claims occur only in historical design/plan documents that are clearly labeled superseded; active specification and README use the Phase 4.1 contract.

- [x] **Step 3: Run the complete quality suite**

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\mypy.exe src
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing --cov-fail-under=90 -q
```

Expected: all tests pass, Ruff and mypy are clean, and coverage is at least 90%.

- [x] **Step 4: Run real headless MetaDrive checks**

```powershell
.venv\Scripts\python.exe -m pytest tests/integration/test_metadrive_headless.py tests/integration/test_control_metadrive_headless.py tests/integration/test_rl_metadrive_headless.py -q
```

Expected: all real integration tests pass with no leaked simulator process.

- [x] **Step 5: Run two reproducible PPO smoke trainings in fresh directories**

> **Historical completed commands:** These destinations are occupied and must not be reused. The review-remediation replacement runs use the new `phase4_1_seed_artifact_smoke_20260721_a` and `_b` destinations recorded below.

```powershell
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --smoke --run-dir runs/phase4_1_smoke_seed42_a
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --smoke --run-dir runs/phase4_1_smoke_seed42_b
```

Expected: both runs complete, each observes at least two distinct episode RNG seeds, both seed sequences match, validation seeds are outside the train range, observations/rewards are finite, and run metadata contains contract version 2.

- [x] **Step 6: Record measured results and deferred observation work**

Append exact test count, warnings, coverage, smoke requested/actual timesteps, elapsed time, observed train/validation seed sequences, checkpoint reload result, and process cleanup result to `docs/phase4_implementation_log.md`. Record explicit validity features (`ttc_valid`, `claim_valid`, `agent_failed`, `target_actor_present`) as deferred and unimplemented.

- [x] **Step 7: Commit documentation and verification evidence**

```powershell
git add README.md docs configs
git commit -m "docs: record phase 4 research hardening"
```

- [x] **Step 8: Perform final diff and repository review**

```powershell
git status --short
git diff --check origin/feat/phase4-rl-environment...HEAD
git log --oneline origin/feat/phase4-rl-environment..HEAD
```

Expected: no unintended files, no whitespace errors, and only Phase 4.1 commits.

Review remediation added strict TDD coverage for actual reset persistence, train/validation separation, collision-free multi-worker paths, malformed-info fail-closed behavior, cloudpickle-safe construction, source/destination ownership, close-before-publish ordering, and optional inventory compatibility for existing version-2 metadata. The old post-training regenerated seed evidence is superseded by the two persisted-artifact smoke runs documented in `docs/phase4_implementation_log.md`.

- [ ] **Step 9: Push and verify Draft PR #4 (delegated to parent; intentionally not run here)**

```powershell
git push origin feat/phase4-rl-environment
gh pr view 4 --repo kazu02210679/drive --json number,url,isDraft,baseRefName,headRefName,title,state,mergeStateStatus
```

Expected: PR #4 remains open and Draft, head is `feat/phase4-rl-environment`, and the latest Phase 4.1 commit is present.
