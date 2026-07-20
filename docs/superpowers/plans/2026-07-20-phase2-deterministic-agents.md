# Phase 2 Deterministic Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic Nominal, Hazard, Rule, and Critic agents that analyze every fixed-action MetaDrive decision step without controlling the vehicle.

**Architecture:** Immutable `SceneSnapshot` remains the only runtime input to the agents. Three pure claim-producing agents share only finite kinematic helpers; a stateless suite invokes them in fixed order and gives their claims to Critic once. The existing smoke runner owns MetaDrive lifecycle and emits only the final snapshot, three claims, and review.

**Tech Stack:** Python 3.11.9, MetaDrive 0.4.3, Gymnasium 1.3.0, Pydantic 2.11.7, NumPy 1.26.4, pytest 8.4.1, Ruff 0.12.4, mypy 1.16.1.

## Global Constraints

- `docs/multi_agent_driving_mvp_spec.md` remains the highest-level requirement; the approved design is `docs/superpowers/specs/2026-07-20-phase2-deterministic-agents-design.md`.
- Add only Phase 2 deterministic analysis. Do not implement action mapping, PID, lane keeping, Safety Shield, Coordinator observation, PPO, scenarios, curriculum, or evaluation reports.
- Nominal, Hazard, and Rule receive the same immutable `SceneSnapshot`; Critic receives the snapshot and the three claims and never reruns an agent.
- All agents are stateless and side-effect free. Claim IDs, actor selection, claim order, reason order, and repeated output must be deterministic.
- Reject non-finite and out-of-range configuration, snapshot, claim, and kinematic input at the nearest boundary.
- Keep Gymnasium five-element `step` behavior, fixed vehicle action, headless default, CPU compatibility, and guaranteed `close()` unchanged.
- Use TDD for every production behavior: write a focused test, observe the intended failure, implement the minimum, and rerun the focused plus relevant regression tests.
- Preserve Python `>=3.11,<3.12`, pinned direct dependencies, `uv.lock`, minimum 80% branch coverage, Ruff, and mypy strict.
- Work on `feat/phase2-deterministic-agents`, based on Phase 1 commit `e826c95`; do not modify Phase 1 PR #1.

## File map

- `src/mad_driving/config/models.py`: strict frozen agent configuration.
- `src/mad_driving/interfaces/scene_snapshot.py`: approved rule-state booleans.
- `src/mad_driving/world_model/snapshot_builder.py`: simulator and scenario values projected into the snapshot.
- `src/mad_driving/agents/protocol.py`: common claim-producing agent contract.
- `src/mad_driving/agents/kinematics.py`: shared finite physics functions only.
- `src/mad_driving/agents/claim_factory.py`: deterministic IDs and neutral claim construction.
- `src/mad_driving/agents/nominal.py`: constant-acceleration actor prediction.
- `src/mad_driving/agents/hazard.py`: worst-case braking, crossing, and occlusion analysis.
- `src/mad_driving/agents/rule.py`: speed and hard-stop constraints.
- `src/mad_driving/agents/critic.py`: eight fixed cross-review checks.
- `src/mad_driving/agents/suite.py`: fixed one-pass orchestration.
- `src/mad_driving/cli/smoke.py`: passive per-step suite invocation and final JSON output.
- `src/mad_driving/envs/multi_agent_speed_env.py`: extended immutable smoke result.

---

### Task 1: Strict agent configuration

**Files:**
- Modify: `src/mad_driving/config/models.py`
- Modify: `configs/base.yaml`
- Create: `tests/unit/config/test_agent_config.py`

**Interfaces:**
- Produces: `NominalAgentConfig`, `HazardAgentConfig`, `RuleAgentConfig`, `CriticAgentConfig`, and `AgentsConfig`.
- Produces: `AppConfig.agents: AgentsConfig`.
- Defaults: horizon `5.0`, step `0.25`, lead deceleration `-8.0`, crossing speed `8.0`, reaction `0.5`, ego deceleration `-6.0`, crawl speed `2.0`, crossing allowance `1.0`, speed spread `5.0`, low confidence `0.5`.

- [x] **Step 1: Write failing strict-config tests**

Create tests that demonstrate the intended API and constraints:

```python
def test_agent_defaults_are_loaded(tmp_path: Path) -> None:
    config = load_config(write_phase2_config(tmp_path))
    assert config.agents.nominal.horizon_s == 5.0
    assert config.agents.nominal.time_step_s == 0.25
    assert config.agents.hazard.lead_max_deceleration_mps2 == -8.0
    assert config.agents.hazard.ego_max_safe_deceleration_mps2 == -6.0
    assert config.agents.critic.recommendation_spread_mps == 5.0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("nominal.horizon_s", 0.0),
        ("nominal.time_step_s", 6.0),
        ("hazard.lead_max_deceleration_mps2", 0.0),
        ("hazard.ego_max_safe_deceleration_mps2", 1.0),
        ("hazard.crossing_actor_max_speed_mps", float("nan")),
        ("critic.low_confidence_threshold", 1.1),
        ("critic.recommendation_spread_mps", -1.0),
    ],
)
def test_invalid_agent_config_is_rejected(path: str, value: float) -> None:
    payload = valid_phase2_payload()
    set_nested(payload["agents"], path, value)
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_unknown_agent_key_is_rejected() -> None:
    payload = valid_phase2_payload()
    payload["agents"]["nominal"]["unknown"] = 1
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AppConfig.model_validate(payload)
```

- [x] **Step 2: Run the tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\config\test_agent_config.py -v`

Expected: collection fails because the agent config classes and `AppConfig.agents` do not exist.

- [x] **Step 3: Implement frozen Pydantic models and explicit YAML defaults**

Use `FiniteFloat` plus `Field` constraints. Add a model validator to Nominal:

```python
class NominalAgentConfig(StrictFrozenModel):
    horizon_s: FiniteFloat = Field(default=5.0, gt=0.0)
    time_step_s: FiniteFloat = Field(default=0.25, gt=0.0)
    ego_length_m: FiniteFloat = Field(default=4.5, gt=0.0)
    ego_width_m: FiniteFloat = Field(default=1.8, gt=0.0)
    lane_half_width_m: FiniteFloat = Field(default=1.75, gt=0.0)
    longitudinal_buffer_m: FiniteFloat = Field(default=0.5, ge=0.0)
    lateral_buffer_m: FiniteFloat = Field(default=0.25, ge=0.0)
    probability_ttc_scale_s: FiniteFloat = Field(default=3.0, gt=0.0)
    probability_distance_scale_m: FiniteFloat = Field(default=3.0, gt=0.0)

    @model_validator(mode="after")
    def validate_time_step(self) -> Self:
        if self.time_step_s > self.horizon_s:
            raise ValueError("time_step_s must not exceed horizon_s")
        return self
```

Implement Hazard and Critic with the exact approved fields and constraints; `RuleAgentConfig` is a strict empty model. Declare `AppConfig.agents` with `Field(default_factory=AgentsConfig)` so old minimal configs continue to load, while adding all four sections explicitly to `configs/base.yaml` so the active defaults remain reviewable.

- [x] **Step 4: Verify GREEN and config regressions**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\config -v`

Expected: all Phase 1 and Phase 2 config tests pass.

- [x] **Step 5: Commit the independently reviewable config boundary**

```powershell
git add configs\base.yaml src\mad_driving\config\models.py tests\unit\config
git commit -m "feat: add deterministic agent configuration"
```

---

### Task 2: Rule-state snapshot boundary

**Files:**
- Modify: `src/mad_driving/interfaces/scene_snapshot.py`
- Modify: `src/mad_driving/world_model/snapshot_builder.py`
- Modify: `tests/unit/interfaces/test_models.py`
- Modify: `tests/unit/world_model/test_snapshot_builder.py`
- Modify: all test snapshot factories under `tests/`

**Interfaces:**
- Adds required `SceneSnapshot.collision_occurred`, `off_road`, and `intersection_entry_prohibited` booleans.
- Extends `SceneSnapshotBuilder.build(..., stop_required=False, occlusion_present=False, distance_to_conflict_point_m=None, intersection_entry_prohibited=False)`.
- Leaves collision and off-road extraction in Task 10 after installed MetaDrive property inspection.

- [x] **Step 1: Add failing interface and builder tests**

```python
def test_scene_snapshot_contains_explicit_rule_state() -> None:
    snapshot = make_snapshot(
        collision_occurred=True,
        off_road=True,
        intersection_entry_prohibited=True,
    )
    assert snapshot.collision_occurred is True
    assert snapshot.off_road is True
    assert snapshot.intersection_entry_prohibited is True


def test_builder_accepts_scenario_flags() -> None:
    snapshot = SceneSnapshotBuilder().build(
        make_env(),
        step_index=1,
        scenario_id="unit",
        seed=42,
        previous_action=0,
        previous_shield_intervention=False,
        stop_required=True,
        occlusion_present=True,
        distance_to_conflict_point_m=12.0,
        intersection_entry_prohibited=True,
    )
    assert snapshot.stop_required is True
    assert snapshot.occlusion_present is True
    assert snapshot.distance_to_conflict_point_m == 12.0
    assert snapshot.intersection_entry_prohibited is True
```

- [x] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\interfaces tests\unit\world_model -v`

Expected: constructor and builder reject the new keyword arguments.

- [x] **Step 3: Add the required fields and keyword-only builder inputs**

Do not add defaults to the dataclass fields; update every constructor so missing rule state is visible in code review. Builder-created `collision_occurred` and `off_road` remain neutral `False` until Task 10 inspects and tests MetaDrive's concrete properties.

- [x] **Step 4: Verify GREEN across all current tests**

Run: `.venv\Scripts\python.exe -m pytest tests\unit -q`

Expected: all tests pass with explicit booleans present.

- [x] **Step 5: Commit the approved boundary change**

```powershell
git add src\mad_driving\interfaces src\mad_driving\world_model tests
git commit -m "feat: add rule state to scene snapshots"
```

---

### Task 3: Finite kinematic primitives

**Files:**
- Create: `src/mad_driving/agents/__init__.py`
- Create: `src/mad_driving/agents/kinematics.py`
- Create: `tests/unit/agents/test_kinematics.py`

**Interfaces:**
- Produces: `sample_times(horizon_s: float, step_s: float) -> tuple[float, ...]`.
- Produces: `project_vector(vector_xy: tuple[float, float], heading_rad: float) -> tuple[float, float]`.
- Produces: `relative_position(initial_xy, relative_velocity_xy, relative_acceleration_xy, time_s) -> tuple[float, float]`.
- Produces: `rectangular_clearance(relative_xy, longitudinal_envelope_m, lateral_envelope_m) -> float`.
- Produces: `stopping_distance(speed_mps, deceleration_mps2) -> float` where deceleration is strictly negative.
- Produces: `safe_speed_for_distance(distance_m, reaction_s, deceleration_mps2) -> float`.

- [x] **Step 1: Write focused failing physics tests**

```python
def test_sample_times_includes_exact_horizon() -> None:
    assert sample_times(1.0, 0.25) == (0.25, 0.5, 0.75, 1.0)


def test_rectangular_clearance_is_zero_inside_envelope() -> None:
    assert rectangular_clearance((2.0, 0.5), 2.5, 1.0) == 0.0


def test_stopping_distance_and_inverse_safe_speed_agree() -> None:
    distance = stopping_distance(12.0, -6.0)
    assert safe_speed_for_distance(distance, 0.0, -6.0) == pytest.approx(12.0)


@pytest.mark.parametrize("bad", [0.0, 1.0, math.nan, math.inf])
def test_stopping_distance_rejects_invalid_deceleration(bad: float) -> None:
    with pytest.raises(ValueError):
        stopping_distance(5.0, bad)
```

Also cover zero speed, negative distance clamped to zero safe speed, vector rotation at `pi/2`, relative constant acceleration, and non-finite inputs.

- [x] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_kinematics.py -v`

Expected: import failure because `mad_driving.agents.kinematics` does not exist.

- [x] **Step 3: Implement only the tested finite formulas**

Use existing `finite_float` validation. Construct sample times by integer index through `floor(horizon/step)` and append the exact horizon only when the last regular sample is smaller; never replace or duplicate a regular sample. Use the approved quadratic positive root:

```python
braking = abs(deceleration_mps2)
return max(0.0, sqrt((braking * reaction_s) ** 2 + 2 * braking * distance_m) - braking * reaction_s)
```

- [x] **Step 4: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_kinematics.py -v`

Expected: every kinematic boundary test passes.

- [x] **Step 5: Commit pure physics utilities**

```powershell
git add src\mad_driving\agents tests\unit\agents\test_kinematics.py
git commit -m "feat: add finite driving kinematics"
```

---

### Task 4: Agent protocol and deterministic claim factory

**Files:**
- Create: `src/mad_driving/agents/protocol.py`
- Create: `src/mad_driving/agents/claim_factory.py`
- Create: `tests/unit/agents/factories.py`
- Create: `tests/unit/agents/test_claim_factory.py`

**Interfaces:**
- Produces: runtime-checkable `DrivingAgent` protocol.
- Produces: `claim_id(agent_id, snapshot, event_type, target_actor_id) -> str`.
- Produces: `neutral_claim(agent_id, snapshot, event_type="no_hazard") -> RiskClaim`.

- [x] **Step 1: Write failing identity and neutral-claim tests**

```python
def test_claim_id_is_stable_and_contains_no_uuid() -> None:
    snapshot = make_snapshot(step_index=7)
    first = claim_id("nominal", snapshot, "nominal_collision", "actor-2")
    second = claim_id("nominal", snapshot, "nominal_collision", "actor-2")
    assert first == second == "nominal:7:actor-2:nominal_collision"


def test_neutral_claim_is_finite_and_valid_for_current_step() -> None:
    snapshot = make_snapshot(speed_limit_mps=13.0)
    claim = neutral_claim("hazard", snapshot)
    assert claim.target_actor_id is None
    assert claim.severity == 0.0
    assert claim.recommended_max_speed_mps == 13.0
    assert claim.evidence
    assert claim.valid_until_step == snapshot.step_index
```

- [x] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_claim_factory.py -v`

Expected: import failure for the new modules.

- [x] **Step 3: Implement the exact approved ID and neutral values**

Use `target_actor_id or "none"` in IDs. Neutral claims use probability `0.0`, confidence `1.0`, severity `0.0`, horizon `0.0`, no TTC or stopping margin, no hard stop, evidence `("no_applicable_hazard",)`, and an empty assumptions tuple.

- [x] **Step 4: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_claim_factory.py -v`

Expected: stable equality and JSON serialization pass.

- [x] **Step 5: Commit contracts**

```powershell
git add src\mad_driving\agents tests\unit\agents\test_claim_factory.py
git commit -m "feat: define deterministic agent claims"
```

---

### Task 5: NominalMotionAgent

**Files:**
- Create: `src/mad_driving/agents/nominal.py`
- Create: `tests/unit/agents/test_nominal.py`
- Modify: `tests/unit/agents/factories.py`

**Interfaces:**
- Consumes: `NominalAgentConfig`, `SceneSnapshot`, kinematic helpers, claim factory.
- Produces: `NominalMotionAgent(config).analyze(snapshot) -> RiskClaim` with `agent_id == "nominal"`.

- [x] **Step 1: Write failing behavior tests using real immutable snapshots**

Cover one behavior per test:

```python
def test_nominal_selects_closing_same_lane_actor() -> None:
    snapshot = make_snapshot(
        ego_speed_mps=10.0,
        actors=(make_actor("lead", longitudinal_m=15.0, longitudinal_speed_mps=2.0),),
    )
    claim = NominalMotionAgent(NominalAgentConfig()).analyze(snapshot)
    assert claim.target_actor_id == "lead"
    assert claim.min_ttc_s is not None
    assert claim.probability is not None and claim.probability > 0.0


def test_nominal_detects_predicted_cut_in() -> None:
    actor = make_actor(
        "cut-in",
        longitudinal_m=10.0,
        lateral_m=3.0,
        lateral_speed_mps=-1.0,
        same_lane=False,
    )
    assert NominalMotionAgent(NominalAgentConfig()).analyze(
        make_snapshot(actors=(actor,))
    ).target_actor_id == "cut-in"


def test_nominal_is_exactly_deterministic() -> None:
    agent = NominalMotionAgent(NominalAgentConfig())
    snapshot = hazardous_snapshot()
    assert agent.analyze(snapshot) == agent.analyze(snapshot)
```

Also test crossing actor inclusion, actor behind exclusion, no-actor neutral claim, stable actor-ID tie-break, non-empty evidence, finite ranges, and configured five-second/0.25-second sampling.

- [x] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_nominal.py -v`

Expected: import failure for `NominalMotionAgent`.

- [x] **Step 3: Implement approved constant-acceleration evaluation**

Project actor velocity and acceleration into ego heading, subtract ego longitudinal motion, sample the approved interval, calculate rectangular clearance/TTC, and calculate:

```python
ttc_term = exp(-ttc / config.probability_ttc_scale_s) if ttc is not None else 0.0
distance_term = exp(-minimum_clearance / config.probability_distance_scale_m)
closing_term = clip(closing_speed / 10.0, 0.0, 1.0)
probability = clip(0.65 * ttc_term + 0.25 * distance_term + 0.10 * closing_term, 0.0, 1.0)
recommended = snapshot.ego.speed_limit_mps * (1.0 - 0.75 * probability)
```

Select by descending severity, descending probability, ascending clearance, then actor ID. Emit one claim only.

- [x] **Step 4: Verify GREEN and focused regressions**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_nominal.py tests\unit\agents\test_kinematics.py -v`

Expected: all Nominal and physics tests pass.

- [x] **Step 5: Commit Nominal independently**

```powershell
git add src\mad_driving\agents tests\unit\agents
git commit -m "feat: add nominal motion agent"
```

---

### Task 6: HazardAgent

**Files:**
- Create: `src/mad_driving/agents/hazard.py`
- Create: `tests/unit/agents/test_hazard.py`

**Interfaces:**
- Consumes: `HazardAgentConfig`, snapshot actors/conflict flags, stopping helpers.
- Produces: `HazardAgent(config).analyze(snapshot) -> RiskClaim` with finite worst-case TTC and margin when geometry exists.

- [ ] **Step 1: Write failing lead, crossing, and occlusion tests**

```python
def test_hazard_reports_negative_margin_for_close_braking_lead() -> None:
    claim = HazardAgent(HazardAgentConfig()).analyze(close_lead_snapshot())
    assert claim.target_actor_id == "lead"
    assert claim.stopping_margin_m is not None
    assert claim.stopping_margin_m < 0.0
    assert claim.severity > 0.5
    assert claim.recommended_max_speed_mps < close_lead_snapshot().ego.speed_mps


def test_hazard_creates_virtual_occlusion_claim() -> None:
    snapshot = make_snapshot(
        occlusion_present=True,
        distance_to_conflict_point_m=10.0,
    )
    claim = HazardAgent(HazardAgentConfig()).analyze(snapshot)
    assert claim.target_actor_id is None
    assert claim.event_type == "occlusion_hazard"
    assert claim.min_ttc_s is not None


def test_occlusion_without_conflict_distance_uses_crawl_speed() -> None:
    claim = HazardAgent(HazardAgentConfig()).analyze(
        make_snapshot(occlusion_present=True, distance_to_conflict_point_m=None)
    )
    assert claim.stopping_margin_m is None
    assert claim.severity == 0.7
    assert claim.recommended_max_speed_mps == 2.0
```

Also test positive margin, crossing earliest-arrival eligibility, observed crossing speed above configured maximum, candidate tie-break, no hazard neutral result, no hard stop, and repeated equality.

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_hazard.py -v`

Expected: import failure for `HazardAgent`.

- [ ] **Step 3: Implement the approved worst-case formulas**

Implement bumper gap, lead stopping distance, reaction plus ego braking distance, quadratic safe speed, logistic severity `1/(1+exp(margin/scale))`, crossing arrival test, and virtual occlusion. Select by descending severity, lower finite TTC, then target actor ID. Keep `hard_stop_required=False` in all Hazard claims.

- [ ] **Step 4: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_hazard.py tests\unit\agents\test_kinematics.py -v`

Expected: all Hazard and physics tests pass.

- [ ] **Step 5: Commit Hazard independently**

```powershell
git add src\mad_driving\agents tests\unit\agents\test_hazard.py
git commit -m "feat: add worst-case hazard agent"
```

---

### Task 7: RuleAgent

**Files:**
- Create: `src/mad_driving/agents/rule.py`
- Create: `tests/unit/agents/test_rule.py`

**Interfaces:**
- Produces: `RuleAgent(config).analyze(snapshot) -> RiskClaim`.
- Hard-stop priority: collision, off-road, intersection prohibition, scenario stop.

- [ ] **Step 1: Write failing rule-table tests**

```python
@pytest.mark.parametrize(
    ("field", "event_type"),
    [
        ("collision_occurred", "collision_stop"),
        ("off_road", "off_road_stop"),
        ("intersection_entry_prohibited", "intersection_stop"),
        ("stop_required", "scenario_stop"),
    ],
)
def test_rule_hard_stops_for_explicit_constraints(field: str, event_type: str) -> None:
    claim = RuleAgent(RuleAgentConfig()).analyze(make_snapshot(**{field: True}))
    assert claim.event_type == event_type
    assert claim.hard_stop_required is True
    assert claim.recommended_max_speed_mps == 0.0


def test_collision_has_priority_when_multiple_rules_apply() -> None:
    claim = RuleAgent(RuleAgentConfig()).analyze(
        make_snapshot(collision_occurred=True, off_road=True, stop_required=True)
    )
    assert claim.event_type == "collision_stop"


def test_normal_rule_claim_recommends_speed_limit() -> None:
    claim = RuleAgent(RuleAgentConfig()).analyze(make_snapshot(speed_limit_mps=12.0))
    assert claim.event_type == "speed_limit"
    assert claim.recommended_max_speed_mps == 12.0
    assert claim.hard_stop_required is False
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_rule.py -v`

Expected: import failure for `RuleAgent`.

- [ ] **Step 3: Implement the fixed priority and overspeed severity**

Use the first true item from the exact priority tuple. Otherwise set probability to `1.0` only when over the speed limit and severity to `clip((speed-limit)/max(limit, 1.0), 0, 1)`.

- [ ] **Step 4: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_rule.py -v`

Expected: all priority and speed-limit tests pass.

- [ ] **Step 5: Commit Rule independently**

```powershell
git add src\mad_driving\agents tests\unit\agents\test_rule.py
git commit -m "feat: add deterministic rule agent"
```

---

### Task 8: CriticAgent eight-rule cross review

**Files:**
- Create: `src/mad_driving/agents/critic.py`
- Create: `tests/unit/agents/test_critic.py`

**Interfaces:**
- Produces: `CriticAgent(config).review(snapshot, claims: Sequence[RiskClaim]) -> CriticReview`.
- Produces reason codes in the exact approved order and never calls an agent.

- [ ] **Step 1: Write one failing test per required review rule**

Use valid claims for rules 1–7 and a deliberately corrupted frozen claim for rule 8:

```python
def test_critic_finds_nominal_hazard_disagreement() -> None:
    review = critic().review(
        make_snapshot(),
        (nominal_claim(severity=0.1), hazard_claim(stopping_margin_m=-1.0), rule_claim()),
    )
    assert "nominal_hazard_disagreement" in review.reasons


def test_critic_finds_all_eight_rules_in_fixed_order() -> None:
    review = critic().review(conflicting_snapshot(), claims_triggering_all_rules())
    assert review.reasons == (
        "nominal_hazard_disagreement",
        "occlusion_underestimated",
        "hard_stop_conflict",
        "speed_recommendation_spread",
        "claim_expired",
        "low_confidence_definitive",
        "missing_evidence",
        "invalid_claim",
    )
    assert review.conflict_score == 1.0


def test_invalid_claim_never_enters_review_arithmetic() -> None:
    claim = nominal_claim()
    object.__setattr__(claim, "severity", math.nan)
    review = critic().review(make_snapshot(), (claim,))
    assert review.max_severity == 1.0
    assert review.reasons == ("invalid_claim",)
```

Also test each reason in isolation, expired boundary equality, duplicate-free challenged IDs, supported agent sorting, empty claims, and low-confidence definitive thresholds.

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_critic.py -v`

Expected: import failure for `CriticAgent`.

- [ ] **Step 3: Implement defensive validation and fixed checks**

First partition claims into valid and invalid without performing arithmetic on invalid fields. Evaluate rules in an ordered tuple. Add challenge IDs in input order with a seen set. Use `len(distinct_reasons)/8.0`, `bool(reasons)`, and maximum valid severity unless any invalid claim forces `1.0`.

- [ ] **Step 4: Verify GREEN and model regressions**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_critic.py tests\unit\interfaces -v`

Expected: all eight rules and existing interface validations pass.

- [ ] **Step 5: Commit Critic independently**

```powershell
git add src\mad_driving\agents tests\unit\agents\test_critic.py
git commit -m "feat: add one-pass critic agent"
```

---

### Task 9: AgentSuite and passive smoke analysis

**Files:**
- Create: `src/mad_driving/agents/suite.py`
- Modify: `src/mad_driving/agents/__init__.py`
- Modify: `src/mad_driving/envs/multi_agent_speed_env.py`
- Modify: `src/mad_driving/cli/smoke.py`
- Create: `tests/unit/agents/test_suite.py`
- Modify: `tests/unit/cli/test_smoke.py`

**Interfaces:**
- Produces: `AgentSuite.from_config(config: AgentsConfig) -> AgentSuite`.
- Produces: `AgentSuite.analyze(snapshot) -> tuple[tuple[RiskClaim, ...], CriticReview]`.
- Extends `SmokeResult` with `final_claims` and `final_review`.
- Extends `run_smoke(..., suite_factory=AgentSuite.from_config)` for unit-test injection while default behavior runs the real suite.

- [ ] **Step 1: Write failing fixed-order suite tests**

```python
def test_suite_returns_three_claims_in_fixed_order_and_one_review() -> None:
    suite = AgentSuite.from_config(AgentsConfig())
    claims, review = suite.analyze(make_snapshot())
    assert tuple(claim.agent_id for claim in claims) == ("nominal", "hazard", "rule")
    assert isinstance(review, CriticReview)


def test_suite_is_stateless_and_deterministic() -> None:
    suite = AgentSuite.from_config(AgentsConfig())
    snapshot = hazardous_snapshot()
    assert suite.analyze(snapshot) == suite.analyze(snapshot)
```

Use small protocol fakes in a separate test to count exactly one call per claim agent and one Critic review without mocking MetaDrive.

- [ ] **Step 2: Write failing smoke integration-unit tests**

Extend the existing fake-environment test:

```python
result = run_smoke(make_config(), env_factory=factory)
assert len(result.final_claims) == 3
assert tuple(c.agent_id for c in result.final_claims) == ("nominal", "hazard", "rule")
assert isinstance(result.final_review, CriticReview)
assert env.actions == [(0.0, 0.25), (0.0, 0.25)]
assert env.closed is True
```

Add an analysis-failure test proving `close()` still runs and a `main` test proving claims/review serialize as finite JSON.

- [ ] **Step 3: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents\test_suite.py tests\unit\cli\test_smoke.py -v`

Expected: suite imports and new SmokeResult fields fail.

- [ ] **Step 4: Implement one-pass orchestration and final-only output**

Construct agents once before the loop. After every built snapshot, call the suite and overwrite only local `final_claims` and `final_review`. Do not feed claims into `env.step` or modify `fixed_action`. Keep environment closure in the existing `finally`.

- [ ] **Step 5: Verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\agents tests\unit\cli -v`

Expected: suite, smoke output, fixed action, deterministic result, and error closure tests pass.

- [ ] **Step 6: Commit orchestration**

```powershell
git add src\mad_driving\agents src\mad_driving\envs src\mad_driving\cli tests\unit
git commit -m "feat: analyze smoke steps with agent suite"
```

---

### Task 10: Installed MetaDrive rule-state mapping and full verification

**Files:**
- Modify: `src/mad_driving/world_model/snapshot_builder.py`
- Modify: `tests/integration/test_metadrive_headless.py`
- Modify: `README.md`
- Create: `docs/phase2_implementation_log.md`
- Modify: `docs/superpowers/plans/2026-07-20-phase2-deterministic-agents.md`

**Interfaces:**
- Verifies installed MetaDrive 0.4.3 ego collision and lane properties before binding names.
- Verifies real reset/step/snapshot/three claims/review/close under headless execution.
- Documents exact commands, API observations, deviations, and verification evidence.

- [ ] **Step 1: Inspect installed MetaDrive state properties read-only**

Run focused `rg` against `.venv/Lib/site-packages/metadrive` for `crash_vehicle`, `crash_object`, `crash_sidewalk`, `on_lane`, and off-road termination logic. Record the exact observed semantics in `docs/phase2_implementation_log.md` before changing the builder.

- [ ] **Step 2: Add a failing fake and real integration assertion**

The fake ego exposes the inspected flags; assert collision is true when any collision flag is true and off-road is true only when the inspected lane property says the ego is outside the drivable lane. Extend the real integration test:

```python
claims, review = AgentSuite.from_config(config.agents).analyze(snapshot)
assert tuple(c.agent_id for c in claims) == ("nominal", "hazard", "rule")
assert_finite_tree(asdict(snapshot))
assert_finite_tree([asdict(claim) for claim in claims])
assert_finite_tree(asdict(review))
```

- [ ] **Step 3: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\world_model tests\integration\test_metadrive_headless.py -v`

Expected: the new fake assertion fails because the builder still emits neutral collision/off-road values.

- [ ] **Step 4: Apply only the documented API-compatible mapping**

Use the inspected 0.4.3 property names with conservative `getattr(..., False)` fallbacks. Do not infer off-road from lane offset without lane width. If the concrete API cannot provide a property, retain `False`, document the limitation and exact reason, and keep its scenario keyword boundary available.

- [ ] **Step 5: Run targeted tests and the real 100-step smoke**

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\world_model tests\integration\test_metadrive_headless.py -v
.venv\Scripts\uv.exe sync --no-editable --group dev
.venv\Scripts\python.exe -m mad_driving.cli.smoke --config configs\base.yaml
```

Expected: integration exits zero; smoke completes 100 decision steps unless terminated/truncated, opens no window, and outputs exactly three final claims plus one review.

- [ ] **Step 6: Update README and implementation evidence**

Add the Phase 2 passive-analysis scope, output shape, smoke command, and explicit Phase 3 exclusions. Record dependency versions, test count, coverage, static checks, simulated seconds, upstream warnings, and every MetaDrive mismatch in `docs/phase2_implementation_log.md`.

- [ ] **Step 7: Run the complete quality gate**

```powershell
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
git diff --check feat/phase1-foundation...HEAD
```

Expected: all tests pass, branch coverage is at least 80%, Ruff and formatting pass, mypy reports no issues, and diff check exits zero. The only accepted warning is the already documented upstream Matplotlib/pyparsing deprecation set.

- [ ] **Step 8: Review scope and commit Phase 2 evidence**

Search for placeholders and forbidden Phase 3+ code. Confirm `env.step` still receives only `config.fixed_action` and no agent recommendation controls the vehicle.

```powershell
git add README.md docs src tests configs
git commit -m "test: verify phase 2 agent simulation"
```

- [ ] **Step 9: Publish as a stacked Draft PR after user confirmation**

```powershell
git push -u origin feat/phase2-deterministic-agents
```

Open a Draft PR with base `feat/phase1-foundation`, head `feat/phase2-deterministic-agents`, validation evidence, and a note to retarget to `main` after Phase 1 PR #1 merges.

---

## Self-review record

- **Spec coverage:** Tasks 4–9 implement the common protocol, Nominal, Hazard, Rule, Critic, deterministic equality, and passive per-step integration from specification sections 5–6 and 21. Task 10 covers MetaDrive lifecycle and Phase 2 acceptance evidence.
- **Approved correction coverage:** Task 2 adds all three user-approved snapshot fields; Task 10 maps real simulator state and documents any unavailable API. Task 9 adds the approved final claims/review smoke JSON without control feedback.
- **Scope check:** Coordinator, Safety Shield, action mapping, PID, scenarios, reward, learning, and evaluation remain excluded and receive no placeholder modules.
- **Placeholder scan:** Every task names exact files, interfaces, test commands, expected failures, implementation formulas, success evidence, and commit scope. No deferred implementation marker remains.
- **Type consistency:** `AppConfig.agents` feeds `AgentSuite.from_config`; all claim producers return `RiskClaim`; suite returns `tuple[tuple[RiskClaim, ...], CriticReview]`; `SmokeResult` stores those exact values.
