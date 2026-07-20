# Phase 3 Control and Safety Shield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect deterministic Agent decisions to a four-action MetaDrive control path through a monotone Safety Shield, lane keeping, and bounded speed PID, then prove the complete path in a 100-step headless simulation.

**Architecture:** Keep `RuleBasedCoordinator`, `SafetyShield`, action mapping, PID, and the MetaDrive custom Policy independent. The control smoke loop analyzes the current snapshot, filters the requested discrete action, passes the executed action to the custom Policy, advances MetaDrive, and records one typed trace without introducing the Phase 4 Gymnasium wrapper, observation, reward, or PPO.

**Tech Stack:** Python 3.11.9, MetaDrive 0.4.3, Gymnasium 1.3.0, Pydantic 2.11.7, pytest 8.4.1, Ruff 0.12.4, mypy 1.16.1.

## Global Constraints

- `docs/multi_agent_driving_mvp_spec.md` remains the highest-level requirement.
- Implement only Phase 3: lane keeping, speed PID, four-action mapping, Safety Shield, and rule-based end-to-end control.
- Preserve the Phase 1/2 fixed-action smoke runner and all 145 existing tests.
- Do not add the 24-dimensional Observation, Reward, Gymnasium environment wrapper, PPO, ScenarioManager, JSONL logging, plots, or training dependencies.
- Coordinator and Shield consume typed snapshots and claims; neither may access MetaDrive runtime objects.
- Coordinator never emits steering or throttle/brake.
- In enforce mode, `executed_action >= requested_action` must hold for the safety order `KEEP < SLOW < PREPARE_STOP < STOP`.
- Every new configuration model is strict and frozen, rejects non-finite values, and is represented explicitly in `configs/base.yaml`.
- Use TDD for every production behavior: observe RED before implementation, then GREEN.
- Keep Python `>=3.11,<3.12`, pinned direct dependencies, branch coverage at least 80%, Ruff, and mypy strict.
- Record installed MetaDrive API observations and every required compatibility adjustment before changing an assumed API binding.

---

## File map

```text
src/mad_driving/
├─ config/models.py                         # Phase 3 strict config models
├─ interfaces/shield_result.py              # Immutable Shield decision
├─ interfaces/__init__.py                   # Public ShieldResult export
├─ control/__init__.py                      # Public control exports
├─ control/actions.py                       # DrivingAction safety order
├─ control/action_mapper.py                 # Action/speed-cap conversion
├─ control/pid.py                           # Bounded PID and anti-windup
├─ control/lane_keeping_policy.py           # MetaDrive custom Policy
├─ coordinator/__init__.py                  # Public Coordinator export
├─ coordinator/rule_based.py                # Stateless rule baseline
├─ safety/__init__.py                       # Public Shield export
├─ safety/shield.py                         # Defensive monotone filter
├─ envs/control_metadrive_env.py             # Runtime custom-Policy binding
├─ envs/multi_agent_speed_env.py             # ControlSmokeResult boundary
├─ cli/control_smoke.py                     # End-to-end control runner and CLI
└─ cli/smoke.py                             # Must remain fixed-action behavior

tests/
├─ unit/config/test_control_config.py
├─ unit/control/test_action_mapper.py
├─ unit/control/test_pid.py
├─ unit/control/test_lane_keeping_policy.py
├─ unit/coordinator/test_rule_based.py
├─ unit/safety/test_shield.py
├─ unit/cli/test_control_smoke.py
└─ integration/test_control_metadrive_headless.py
```

---

### Task 1: Strict Phase 3 configuration

**Files:**
- Modify: `src/mad_driving/config/models.py`
- Modify: `configs/base.yaml`
- Create: `tests/unit/config/test_control_config.py`

**Interfaces:**
- Produces: `CoordinatorConfig`, `ShieldConfig`, `SpeedPIDConfig`, `SteeringPIDConfig`, `ControlConfig`.
- Extends: `AppConfig.coordinator`, `AppConfig.shield`, and `AppConfig.control`, each with a default factory.
- Later tasks consume only these frozen models; they do not read YAML or environment variables directly.

- [x] **Step 1: Write failing default and strictness tests**

```python
import math

import pytest
from pydantic import ValidationError

from mad_driving.config.models import (
    AppConfig,
    ControlConfig,
    CoordinatorConfig,
    ShieldConfig,
)


def minimum_app_config() -> dict[str, object]:
    return {
        "seed": 42,
        "scenario_id": "phase3_config",
        "decision_steps": 10,
        "fixed_action": [0.0, 0.25],
        "metadrive": {},
    }


def test_phase3_config_defaults_are_strict_and_frozen() -> None:
    config = AppConfig.model_validate(minimum_app_config())
    assert config.coordinator.severe_threshold == 0.75
    assert config.shield.mode == "enforce"
    assert config.control.speed.emergency_deceleration_mps2 == -6.0
    assert config.control.steering.lookahead_m == 1.0
    with pytest.raises(ValidationError):
        CoordinatorConfig.model_validate({"unknown": 1})
    with pytest.raises(ValidationError):
        config.shield.mode = "off"  # type: ignore[misc]


@pytest.mark.parametrize(
    "payload",
    [
        {"imminent_ttc_s": 4.0, "caution_ttc_s": 3.0},
        {"emergency_margin_m": 6.0, "caution_margin_m": 5.0},
        {"mode": "unknown"},
        {"imminent_ttc_s": math.nan},
    ],
)
def test_shield_config_rejects_invalid_relationships(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ShieldConfig.model_validate(payload)


def test_control_config_rejects_wrong_acceleration_order() -> None:
    with pytest.raises(ValidationError):
        ControlConfig.model_validate(
            {
                "speed": {
                    "normal_deceleration_mps2": -7.0,
                    "emergency_deceleration_mps2": -6.0,
                }
            }
        )
```

- [x] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\config\test_control_config.py -v`

Expected: collection fails because the Phase 3 config classes do not exist.

- [x] **Step 3: Implement exact strict models**

Add these models before `AppConfig` and import `Literal`:

```python
class CoordinatorConfig(StrictFrozenModel):
    conflict_min_action: int = Field(default=1, ge=0, le=3)
    severe_min_action: int = Field(default=2, ge=0, le=3)
    severe_threshold: FiniteFloat = Field(default=0.75, ge=0.0, le=1.0)


class ShieldConfig(StrictFrozenModel):
    mode: Literal["off", "monitor", "enforce"] = "enforce"
    imminent_ttc_s: FiniteFloat = Field(default=1.0, gt=0.0)
    caution_ttc_s: FiniteFloat = Field(default=3.0, gt=0.0)
    emergency_margin_m: FiniteFloat = 0.0
    caution_margin_m: FiniteFloat = 5.0
    missing_agent_action: int = Field(default=2, ge=0, le=3)
    multiple_missing_action: int = Field(default=3, ge=0, le=3)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> Self:
        if self.imminent_ttc_s > self.caution_ttc_s:
            raise ValueError("imminent_ttc_s must not exceed caution_ttc_s")
        if self.emergency_margin_m > self.caution_margin_m:
            raise ValueError("emergency_margin_m must not exceed caution_margin_m")
        return self


class SpeedPIDConfig(StrictFrozenModel):
    kp: FiniteFloat = Field(default=0.50, ge=0.0)
    ki: FiniteFloat = Field(default=0.05, ge=0.0)
    kd: FiniteFloat = Field(default=0.10, ge=0.0)
    integral_limit: FiniteFloat = Field(default=10.0, gt=0.0)
    max_acceleration_mps2: FiniteFloat = Field(default=2.5, gt=0.0)
    normal_deceleration_mps2: FiniteFloat = Field(default=-3.0, lt=0.0)
    emergency_deceleration_mps2: FiniteFloat = Field(default=-6.0, lt=0.0)

    @model_validator(mode="after")
    def validate_deceleration_order(self) -> Self:
        if self.emergency_deceleration_mps2 > self.normal_deceleration_mps2:
            raise ValueError(
                "emergency_deceleration_mps2 must not exceed normal_deceleration_mps2"
            )
        return self


class SteeringPIDConfig(StrictFrozenModel):
    heading_kp: FiniteFloat = Field(default=1.7, ge=0.0)
    heading_ki: FiniteFloat = Field(default=0.01, ge=0.0)
    heading_kd: FiniteFloat = Field(default=3.5, ge=0.0)
    lateral_kp: FiniteFloat = Field(default=0.3, ge=0.0)
    lateral_ki: FiniteFloat = Field(default=0.002, ge=0.0)
    lateral_kd: FiniteFloat = Field(default=0.05, ge=0.0)
    integral_limit: FiniteFloat = Field(default=5.0, gt=0.0)
    lookahead_m: FiniteFloat = Field(default=1.0, gt=0.0)


class ControlConfig(StrictFrozenModel):
    speed: SpeedPIDConfig = Field(default_factory=SpeedPIDConfig)
    steering: SteeringPIDConfig = Field(default_factory=SteeringPIDConfig)
```

Add to `AppConfig`:

```python
coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
shield: ShieldConfig = Field(default_factory=ShieldConfig)
control: ControlConfig = Field(default_factory=ControlConfig)
```

Add the exact three YAML sections from the approved design to `configs/base.yaml`.

- [x] **Step 4: Verify GREEN and existing loader compatibility**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\config -v`

Expected: all new and existing config tests pass; old payloads without Phase 3 sections use defaults.

- [ ] **Step 5: Commit configuration independently**

```powershell
git add src\mad_driving\config\models.py configs\base.yaml tests\unit\config\test_control_config.py
git commit -m "feat: add phase 3 control configuration"
```

---

### Task 2: Four-action model and target-speed mapping

**Files:**
- Create: `src/mad_driving/control/__init__.py`
- Create: `src/mad_driving/control/actions.py`
- Create: `src/mad_driving/control/action_mapper.py`
- Create: `tests/unit/control/test_action_mapper.py`

**Interfaces:**
- Produces: `DrivingAction(IntEnum)` with `KEEP=0`, `SLOW=1`, `PREPARE_STOP=2`, `STOP=3`.
- Produces: `target_speed_mps(action, current_speed_mps, speed_limit_mps) -> float`.
- Produces: `action_for_speed_cap(recommended_max_speed_mps, speed_limit_mps) -> DrivingAction`.

- [ ] **Step 1: Write failing mapping and boundary tests**

```python
import math

import pytest

from mad_driving.control import (
    DrivingAction,
    action_for_speed_cap,
    target_speed_mps,
)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (DrivingAction.KEEP, 20.0),
        (DrivingAction.SLOW, 10.0),
        (DrivingAction.PREPARE_STOP, 5.0),
        (DrivingAction.STOP, 0.0),
    ],
)
def test_target_speed_mapping(action: DrivingAction, expected: float) -> None:
    assert target_speed_mps(action, 10.0, 20.0) == expected


@pytest.mark.parametrize(
    ("recommended", "expected"),
    [
        (0.0, DrivingAction.STOP),
        (5.0, DrivingAction.PREPARE_STOP),
        (5.0001, DrivingAction.SLOW),
        (12.0, DrivingAction.SLOW),
        (12.0001, DrivingAction.KEEP),
    ],
)
def test_speed_cap_boundaries(recommended: float, expected: DrivingAction) -> None:
    assert action_for_speed_cap(recommended, 20.0) is expected


def test_zero_speed_limit_requires_stop() -> None:
    assert action_for_speed_cap(10.0, 0.0) is DrivingAction.STOP


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_mapping_rejects_invalid_speeds(value: float) -> None:
    with pytest.raises(ValueError):
        target_speed_mps(DrivingAction.KEEP, value, 10.0)
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\control\test_action_mapper.py -v`

Expected: import failure for `mad_driving.control`.

- [ ] **Step 3: Implement the pure action boundary**

`actions.py`:

```python
from enum import IntEnum


class DrivingAction(IntEnum):
    KEEP = 0
    SLOW = 1
    PREPARE_STOP = 2
    STOP = 3
```

`action_mapper.py`:

```python
from math import isfinite

from mad_driving.control.actions import DrivingAction


def _speed(name: str, value: float) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def target_speed_mps(
    action: DrivingAction | int,
    current_speed_mps: float,
    speed_limit_mps: float,
) -> float:
    action = DrivingAction(action)
    current = _speed("current_speed_mps", current_speed_mps)
    limit = _speed("speed_limit_mps", speed_limit_mps)
    if action is DrivingAction.KEEP:
        return limit
    if action is DrivingAction.SLOW:
        return min(current, 0.60 * limit)
    if action is DrivingAction.PREPARE_STOP:
        return min(current, 0.25 * limit)
    return 0.0


def action_for_speed_cap(
    recommended_max_speed_mps: float,
    speed_limit_mps: float,
) -> DrivingAction:
    recommended = _speed("recommended_max_speed_mps", recommended_max_speed_mps)
    limit = _speed("speed_limit_mps", speed_limit_mps)
    if limit == 0.0 or recommended <= 0.0:
        return DrivingAction.STOP
    if recommended <= 0.25 * limit:
        return DrivingAction.PREPARE_STOP
    if recommended <= 0.60 * limit:
        return DrivingAction.SLOW
    return DrivingAction.KEEP
```

Export all three public names from `control/__init__.py`.

- [ ] **Step 4: Verify GREEN and exact determinism**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\control\test_action_mapper.py -v`

Expected: all action, boundary, and invalid-input tests pass.

- [ ] **Step 5: Commit the action model**

```powershell
git add src\mad_driving\control tests\unit\control\test_action_mapper.py
git commit -m "feat: add four-action speed mapping"
```

---

### Task 3: Stateless rule-based Coordinator

**Files:**
- Create: `src/mad_driving/coordinator/__init__.py`
- Create: `src/mad_driving/coordinator/rule_based.py`
- Modify: `tests/unit/agents/factories.py`
- Create: `tests/unit/coordinator/test_rule_based.py`

**Interfaces:**
- Consumes: `CoordinatorConfig`, `SceneSnapshot`, `Sequence[RiskClaim]`, `CriticReview`.
- Produces: `RuleBasedCoordinator(config).decide(snapshot, claims, review) -> DrivingAction`.
- Produces test helper: `make_claim(agent_id="nominal", **overrides) -> RiskClaim` in `tests/unit/agents/factories.py`.

- [ ] **Step 1: Add a reusable claim factory and failing Coordinator tests**

Add this test helper with full valid defaults:

```python
def make_claim(agent_id: str = "nominal", **overrides: Any) -> RiskClaim:
    values: dict[str, Any] = {
        "claim_id": f"{agent_id}:1:none:test",
        "agent_id": agent_id,
        "event_type": "test",
        "target_actor_id": None,
        "probability": 0.0,
        "confidence": 1.0,
        "severity": 0.0,
        "time_horizon_s": 1.0,
        "min_ttc_s": None,
        "stopping_margin_m": None,
        "recommended_max_speed_mps": 20.0,
        "hard_stop_required": False,
        "evidence": ("test",),
        "assumptions": (),
        "valid_until_step": 1,
    }
    values.update(overrides)
    return RiskClaim(**values)
```

Create tests:

```python
from mad_driving.config.models import CoordinatorConfig
from mad_driving.coordinator import RuleBasedCoordinator
from mad_driving.control import DrivingAction
from mad_driving.interfaces import CriticReview
from tests.unit.agents.factories import make_claim, make_snapshot


def review(*, unresolved: bool = False, max_severity: float = 0.0) -> CriticReview:
    return CriticReview(
        conflict_score=1.0 if unresolved else 0.0,
        unresolved_conflict=unresolved,
        max_severity=max_severity,
        supported_agent_ids=(),
        challenged_claim_ids=(),
        reasons=("test_conflict",) if unresolved else (),
    )


def test_coordinator_uses_most_restrictive_claim_speed() -> None:
    action = RuleBasedCoordinator(CoordinatorConfig()).decide(
        make_snapshot(speed_limit_mps=20.0),
        (
            make_claim("nominal", recommended_max_speed_mps=20.0),
            make_claim("hazard", recommended_max_speed_mps=5.0),
            make_claim("rule", recommended_max_speed_mps=20.0),
        ),
        review(),
    )
    assert action is DrivingAction.PREPARE_STOP


def test_hard_stop_overrides_every_other_candidate() -> None:
    action = RuleBasedCoordinator(CoordinatorConfig()).decide(
        make_snapshot(),
        (make_claim("rule", hard_stop_required=True),),
        review(),
    )
    assert action is DrivingAction.STOP


def test_any_required_agent_missing_applies_prepare_stop_floor() -> None:
    action = RuleBasedCoordinator(CoordinatorConfig()).decide(
        make_snapshot(),
        (make_claim("nominal"), make_claim("rule")),
        review(),
    )
    assert action is DrivingAction.PREPARE_STOP


def test_conflict_and_severity_apply_minimum_actions() -> None:
    coordinator = RuleBasedCoordinator(CoordinatorConfig())
    assert coordinator.decide(make_snapshot(), (), review(unresolved=True)) is DrivingAction.PREPARE_STOP
    assert coordinator.decide(make_snapshot(), (make_claim(),), review(max_severity=0.75)) is DrivingAction.PREPARE_STOP


def test_identical_input_is_exactly_deterministic() -> None:
    coordinator = RuleBasedCoordinator(CoordinatorConfig())
    arguments = (make_snapshot(), (make_claim(),), review())
    assert coordinator.decide(*arguments) == coordinator.decide(*arguments)
```

The conflict test expects PREPARE_STOP because an empty claim sequence invokes the safe missing-input floor, which is stricter than the configured conflict floor SLOW.

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\coordinator\test_rule_based.py -v`

Expected: import failure for `mad_driving.coordinator`.

- [ ] **Step 3: Implement candidate aggregation without physics duplication**

```python
from collections.abc import Sequence

from mad_driving.config.models import CoordinatorConfig
from mad_driving.control import DrivingAction, action_for_speed_cap
from mad_driving.interfaces import CriticReview, RiskClaim, SceneSnapshot


class RuleBasedCoordinator:
    _required_agent_ids = frozenset({"nominal", "hazard", "rule"})

    def __init__(self, config: CoordinatorConfig) -> None:
        self._config = config

    def decide(
        self,
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> DrivingAction:
        present_agent_ids = {claim.agent_id for claim in claims}
        if self._required_agent_ids - present_agent_ids:
            base = DrivingAction.PREPARE_STOP
        else:
            base = max(
                action_for_speed_cap(
                    claim.recommended_max_speed_mps,
                    snapshot.ego.speed_limit_mps,
                )
                for claim in claims
            )
        candidates = [base]
        if any(claim.hard_stop_required for claim in claims):
            candidates.append(DrivingAction.STOP)
        if review.unresolved_conflict:
            candidates.append(DrivingAction(self._config.conflict_min_action))
        if review.max_severity >= self._config.severe_threshold:
            candidates.append(DrivingAction(self._config.severe_min_action))
        return max(candidates)
```

Do not inspect TTC or stopping margin in this class.

- [ ] **Step 4: Verify GREEN with mapping regressions**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\coordinator tests\unit\control\test_action_mapper.py -v`

Expected: all Coordinator and action mapping tests pass.

- [ ] **Step 5: Commit Coordinator independently**

```powershell
git add src\mad_driving\coordinator tests\unit\coordinator tests\unit\agents\factories.py
git commit -m "feat: add rule-based speed coordinator"
```

---

### Task 4: Defensive monotone Safety Shield

**Files:**
- Create: `src/mad_driving/interfaces/shield_result.py`
- Modify: `src/mad_driving/interfaces/__init__.py`
- Create: `src/mad_driving/safety/__init__.py`
- Create: `src/mad_driving/safety/shield.py`
- Create: `tests/unit/safety/test_shield.py`

**Interfaces:**
- Produces: frozen `ShieldResult` with requested, required, executed, intervention flags, and reasons.
- Produces: `SafetyShield(config).filter(requested_action, snapshot, claims) -> ShieldResult`.
- Consumes only typed inputs and `ShieldConfig`; no MetaDrive access.

- [ ] **Step 1: Write failing mode, reason, and monotonicity tests**

```python
import math

import pytest

from mad_driving.config.models import ShieldConfig
from mad_driving.control import DrivingAction
from mad_driving.safety import SafetyShield
from tests.unit.agents.factories import make_claim, make_snapshot


def test_enforce_never_relaxes_requested_action() -> None:
    shield = SafetyShield(ShieldConfig(mode="enforce"))
    for requested in DrivingAction:
        result = shield.filter(requested, make_snapshot(), (make_claim(),))
        assert result.executed_action >= requested


def test_modes_distinguish_candidate_from_real_intervention() -> None:
    claims = (make_claim("hazard", min_ttc_s=0.5),)
    off = SafetyShield(ShieldConfig(mode="off")).filter(
        DrivingAction.KEEP, make_snapshot(), claims
    )
    monitor = SafetyShield(ShieldConfig(mode="monitor")).filter(
        DrivingAction.KEEP, make_snapshot(), claims
    )
    enforce = SafetyShield(ShieldConfig(mode="enforce")).filter(
        DrivingAction.KEEP, make_snapshot(), claims
    )
    assert off.executed_action is DrivingAction.KEEP
    assert off.reasons == ()
    assert monitor.required_action is DrivingAction.STOP
    assert monitor.intervention_required is True
    assert monitor.intervened is False
    assert enforce.executed_action is DrivingAction.STOP
    assert enforce.intervened is True


def test_all_reasons_have_fixed_duplicate_free_order() -> None:
    claims = (
        make_claim(
            "nominal",
            min_ttc_s=0.5,
            recommended_max_speed_mps=0.0,
            hard_stop_required=True,
        ),
    )
    result = SafetyShield(ShieldConfig()).filter(
        DrivingAction.KEEP,
        make_snapshot(collision_occurred=True, off_road=True),
        claims,
    )
    assert result.reasons == (
        "collision_occurred",
        "off_road",
        "hard_stop_required",
        "multiple_agents_missing",
        "imminent_ttc",
        "claim_speed_limit",
    )


def test_margin_boundaries_are_explicit() -> None:
    shield = SafetyShield(ShieldConfig())
    zero = shield.filter(
        DrivingAction.KEEP,
        make_snapshot(),
        (
            make_claim("nominal"),
            make_claim("hazard", stopping_margin_m=0.0),
            make_claim("rule"),
        ),
    )
    negative = shield.filter(
        DrivingAction.KEEP,
        make_snapshot(),
        (
            make_claim("nominal"),
            make_claim("hazard", stopping_margin_m=-0.001),
            make_claim("rule"),
        ),
    )
    assert "negative_stopping_margin" not in zero.reasons
    assert "low_stopping_margin" in zero.reasons
    assert "negative_stopping_margin" in negative.reasons
    assert negative.executed_action >= zero.executed_action


def test_invalid_claim_is_stopped_before_arithmetic() -> None:
    claim = make_claim()
    object.__setattr__(claim, "min_ttc_s", math.nan)
    result = SafetyShield(ShieldConfig()).filter(
        DrivingAction.KEEP, make_snapshot(), (claim,)
    )
    assert result.executed_action is DrivingAction.STOP
    assert result.reasons == ("invalid_input", "multiple_agents_missing")
```

Also add parameterized tests for every individual reason, one missing versus two missing Agents, exact TTC boundaries, hard stop, collision/off-road, monitor mode, stricter claim caps, and repeated equality.

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\safety\test_shield.py -v`

Expected: imports fail because `ShieldResult` and `SafetyShield` do not exist.

- [ ] **Step 3: Implement `ShieldResult` validation**

```python
from dataclasses import dataclass

from mad_driving.control.actions import DrivingAction


@dataclass(frozen=True)
class ShieldResult:
    requested_action: DrivingAction
    required_action: DrivingAction
    executed_action: DrivingAction
    intervention_required: bool
    intervened: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("requested_action", "required_action", "executed_action"):
            object.__setattr__(self, name, DrivingAction(getattr(self, name)))
        if self.intervention_required != (self.required_action > self.requested_action):
            raise ValueError("intervention_required is inconsistent")
        if self.intervened != (self.executed_action != self.requested_action):
            raise ValueError("intervened is inconsistent")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("reasons must be duplicate-free")
```

- [ ] **Step 4: Implement defensive partitioning and fixed rules**

Implement `SafetyShield` with these private helpers and no arithmetic on invalid claims:

```python
_REQUIRED_AGENT_IDS = frozenset({"nominal", "hazard", "rule"})


def _valid_claim(claim: RiskClaim) -> bool:
    try:
        RiskClaim(**asdict(claim))
    except (TypeError, ValueError):
        return False
    return True


def _valid_snapshot(snapshot: SceneSnapshot) -> bool:
    try:
        values = asdict(snapshot)
        ego = EgoState(**values.pop("ego"))
        actors = tuple(ActorState(**actor) for actor in values.pop("actors"))
        SceneSnapshot(ego=ego, actors=actors, **values)
    except (TypeError, ValueError):
        return False
    return True
```

In `filter`:

```python
requested = DrivingAction(requested_action)
if self._config.mode == "off":
    return ShieldResult(requested, requested, requested, False, False, ())

valid_claims = tuple(claim for claim in claims if _valid_claim(claim))
reasons: list[str] = []
candidates = [DrivingAction.KEEP]

if len(valid_claims) != len(claims) or not _valid_snapshot(snapshot):
    reasons.append("invalid_input")
    candidates.append(DrivingAction.STOP)
if snapshot.collision_occurred:
    reasons.append("collision_occurred")
    candidates.append(DrivingAction.STOP)
if snapshot.off_road:
    reasons.append("off_road")
    candidates.append(DrivingAction.STOP)
if any(claim.hard_stop_required for claim in valid_claims):
    reasons.append("hard_stop_required")
    candidates.append(DrivingAction.STOP)

missing_count = len(_REQUIRED_AGENT_IDS - {claim.agent_id for claim in valid_claims})
if missing_count >= 2:
    reasons.append("multiple_agents_missing")
    candidates.append(DrivingAction(self._config.multiple_missing_action))
elif missing_count == 1:
    reasons.append("agent_missing")
    candidates.append(DrivingAction(self._config.missing_agent_action))
```

Continue with the remaining fixed rules:

```python
ttc_values = tuple(
    claim.min_ttc_s for claim in valid_claims if claim.min_ttc_s is not None
)
minimum_ttc = min(ttc_values, default=None)
if minimum_ttc is not None and minimum_ttc <= self._config.imminent_ttc_s:
    reasons.append("imminent_ttc")
    candidates.append(DrivingAction.STOP)

margin_values = tuple(
    claim.stopping_margin_m
    for claim in valid_claims
    if claim.stopping_margin_m is not None
)
minimum_margin = min(margin_values, default=None)
if (
    minimum_margin is not None
    and minimum_margin < self._config.emergency_margin_m
):
    reasons.append("negative_stopping_margin")
    candidates.append(DrivingAction.STOP)

if (
    minimum_ttc is not None
    and minimum_ttc > self._config.imminent_ttc_s
    and minimum_ttc <= self._config.caution_ttc_s
):
    reasons.append("caution_ttc")
    candidates.append(DrivingAction.PREPARE_STOP)

if (
    minimum_margin is not None
    and minimum_margin >= self._config.emergency_margin_m
    and minimum_margin < self._config.caution_margin_m
):
    reasons.append("low_stopping_margin")
    candidates.append(DrivingAction.PREPARE_STOP)

claim_action = max(
    (
        action_for_speed_cap(
            claim.recommended_max_speed_mps,
            snapshot.ego.speed_limit_mps,
        )
        for claim in valid_claims
    ),
    default=DrivingAction.KEEP,
)
if claim_action > requested:
    reasons.append("claim_speed_limit")
    candidates.append(claim_action)

required = max(candidates)
executed = (
    requested
    if self._config.mode == "monitor"
    else max(requested, required)
)
return ShieldResult(
    requested_action=requested,
    required_action=required,
    executed_action=executed,
    intervention_required=required > requested,
    intervened=executed != requested,
    reasons=tuple(reasons),
)
```

The mutually exclusive emergency/caution comparisons prevent duplicate TTC or margin reasons for the same minimum value. Use `<=` for TTC and `<` for margins exactly as shown.

- [ ] **Step 5: Verify GREEN and interface regressions**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\safety tests\unit\interfaces -v`

Expected: all reason, mode, boundary, invalid-input, monotonicity, and existing interface tests pass.

- [ ] **Step 6: Run a generated monotonicity matrix**

Add and run a test that loops over all four requested actions, TTC values `(None, 5.0, 3.0, 1.0, 0.5)`, and margins `(None, 10.0, 5.0, 0.0, -0.1)`. Assert decreasing TTC or margin never lowers `executed_action` for a fixed requested action.

Run: `.venv\Scripts\python.exe -m pytest tests\unit\safety\test_shield.py -v`

Expected: the complete matrix passes without randomized inputs.

- [ ] **Step 7: Commit Shield independently**

```powershell
git add src\mad_driving\interfaces src\mad_driving\safety tests\unit\safety
git commit -m "feat: add monotone safety shield"
```

---

### Task 5: Bounded PID with conditional anti-windup

**Files:**
- Create: `src/mad_driving/control/pid.py`
- Modify: `src/mad_driving/control/__init__.py`
- Create: `tests/unit/control/test_pid.py`

**Interfaces:**
- Produces: `BoundedPID(kp, ki, kd, integral_limit)`.
- Produces: `update(error, dt_s, lower, upper) -> float`, `reset() -> None`.
- Exposes read-only `integral` and `previous_error` properties for deterministic tests and diagnostics.

- [ ] **Step 1: Write failing P/I/D, saturation, and reset tests**

```python
import math

import pytest

from mad_driving.control import BoundedPID


def test_pid_uses_time_scaled_integral_and_derivative() -> None:
    pid = BoundedPID(kp=1.0, ki=1.0, kd=1.0, integral_limit=10.0)
    first = pid.update(error=2.0, dt_s=0.5, lower=-10.0, upper=10.0)
    second = pid.update(error=3.0, dt_s=0.5, lower=-10.0, upper=10.0)
    assert first == pytest.approx(3.0)
    assert second == pytest.approx(7.5)


def test_upper_saturation_blocks_windup_but_allows_unwinding() -> None:
    pid = BoundedPID(kp=1.0, ki=1.0, kd=0.0, integral_limit=10.0)
    assert pid.update(10.0, 1.0, -1.0, 1.0) == 1.0
    assert pid.integral == 0.0
    pid.update(-0.5, 1.0, -1.0, 1.0)
    assert pid.integral == -0.5


def test_reset_restores_first_update_behavior() -> None:
    pid = BoundedPID(1.0, 1.0, 1.0, 10.0)
    expected = pid.update(2.0, 0.5, -10.0, 10.0)
    pid.update(4.0, 0.5, -10.0, 10.0)
    pid.reset()
    assert pid.update(2.0, 0.5, -10.0, 10.0) == expected


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_pid_rejects_non_finite_values(value: float) -> None:
    pid = BoundedPID(1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        pid.update(value, 0.1, -1.0, 1.0)
```

Also test lower saturation, integral clipping, non-positive `dt_s`, and `lower > upper`.

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\control\test_pid.py -v`

Expected: import failure for `BoundedPID`.

- [ ] **Step 3: Implement conditional integration exactly**

```python
class BoundedPID:
    def update(self, error: float, dt_s: float, lower: float, upper: float) -> float:
        _require_finite(error, dt_s, lower, upper)
        if dt_s <= 0.0 or lower > upper:
            raise ValueError("invalid PID update bounds")
        derivative = (
            0.0
            if self._previous_error is None
            else (error - self._previous_error) / dt_s
        )
        candidate = _clip(
            self._integral + error * dt_s,
            -self._integral_limit,
            self._integral_limit,
        )
        raw = self._kp * error + self._ki * candidate + self._kd * derivative
        saturated_high = raw > upper
        saturated_low = raw < lower
        if not ((saturated_high and error > 0.0) or (saturated_low and error < 0.0)):
            self._integral = candidate
            raw = self._kp * error + self._ki * self._integral + self._kd * derivative
        self._previous_error = error
        return _clip(raw, lower, upper)

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = None
```

Constructor validation rejects negative gains, non-positive integral limit, and non-finite inputs.

- [ ] **Step 4: Verify GREEN and format**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\control\test_pid.py -v`

Expected: all PID state, saturation, anti-windup, reset, and invalid-input tests pass.

- [ ] **Step 5: Commit PID independently**

```powershell
git add src\mad_driving\control tests\unit\control\test_pid.py
git commit -m "feat: add bounded anti-windup pid"
```

---

### Task 6: Lane-keeping longitudinal MetaDrive Policy

**Files:**
- Create: `src/mad_driving/control/lane_keeping_policy.py`
- Modify: `src/mad_driving/control/__init__.py`
- Create: `src/mad_driving/envs/control_metadrive_env.py`
- Modify: `src/mad_driving/envs/__init__.py`
- Create: `tests/unit/control/test_lane_keeping_policy.py`
- Create: `tests/integration/test_control_metadrive_headless.py`
- Create: `docs/phase3_implementation_log.md`

**Interfaces:**
- Produces: `LaneKeepingLongitudinalPolicy(BasePolicy)` with `Discrete(4)` input.
- Produces: `create_control_metadrive_env(config, control_config) -> DrivingEnvironment`.
- Policy reads `control_config` from the MetaDrive global config added by a small `ControlMetaDriveEnv` subclass.

- [ ] **Step 1: Inspect and record the installed API before binding it**

Read only:

```powershell
rg -n "class BasePolicy|def act|agent_policy|external_actions|get_input_space" .venv\Lib\site-packages\metadrive -g "*.py"
rg -n "current_lane|heading_theta_at|local_coordinates|speed_km_h|max_speed_m_s" .venv\Lib\site-packages\metadrive -g "*.py"
```

Record exact MetaDrive 0.4.3 constructor signatures, external action lookup, lane methods, speed units, and custom config handling in `docs/phase3_implementation_log.md` before production code. If the installed API differs from the design, document the mismatch and use the smallest compatible binding.

- [ ] **Step 2: Write failing pure-policy tests with a fake vehicle and lane**

Test the computation through a MetaDrive-independent helper `_compute_action(vehicle, action, dt_s)` created on the Policy class, while a real integration test covers engine lookup.

```python
import math

from mad_driving.config.models import ControlConfig
from mad_driving.control import DrivingAction, LaneKeepingLongitudinalPolicy


def test_policy_keep_accelerates_below_target(fake_policy, fake_vehicle) -> None:
    fake_vehicle.speed = 2.0
    steering, throttle = fake_policy._compute_action(
        fake_vehicle, DrivingAction.KEEP, 0.1
    )
    assert -1.0 <= steering <= 1.0
    assert 0.0 < throttle <= 1.0


def test_policy_stop_uses_emergency_brake(fake_policy, fake_vehicle) -> None:
    fake_vehicle.speed = 10.0
    _, throttle = fake_policy._compute_action(fake_vehicle, DrivingAction.STOP, 0.1)
    assert throttle == -1.0


def test_lateral_and_heading_errors_steer_toward_lane(fake_policy, fake_vehicle) -> None:
    fake_vehicle.position = (5.0, 1.0)
    right_correction = fake_policy._compute_action(
        fake_vehicle, DrivingAction.KEEP, 0.1
    )[0]
    fake_policy.reset_controller_state()
    fake_vehicle.position = (5.0, -1.0)
    left_correction = fake_policy._compute_action(
        fake_vehicle, DrivingAction.KEEP, 0.1
    )[0]
    assert right_correction == -left_correction
    assert right_correction != 0.0


def test_missing_lane_fails_safe(fake_policy, fake_vehicle) -> None:
    fake_vehicle.navigation.current_lane = None
    fake_vehicle.lane = None
    assert fake_policy._compute_action(
        fake_vehicle, DrivingAction.KEEP, 0.1
    ) == (0.0, -1.0)
    assert fake_policy.action_info["fail_safe"] is True
```

The fixture may construct the instance with `object.__new__`, assign validated config and three `BoundedPID` objects, and avoid creating the Panda3D engine. Also test reset, non-finite vehicle speed, invalid action, speed-limit fallback, and finite action_info.

- [ ] **Step 3: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\control\test_lane_keeping_policy.py -v`

Expected: import failure for `LaneKeepingLongitudinalPolicy`.

- [ ] **Step 4: Implement Policy construction and normalized control**

The public class must have these boundaries:

```python
class LaneKeepingLongitudinalPolicy(BasePolicy):
    def __init__(self, control_object: Any, random_seed: int | None = None) -> None:
        super().__init__(control_object=control_object, random_seed=random_seed)
        self._control_config = ControlConfig.model_validate(
            self.engine.global_config["control_config"]
        )
        self._decision_interval_s = decision_interval_s(self.engine.global_config)
        self._build_controllers()

    @classmethod
    def get_input_space(cls) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(4)

    def act(self, agent_id: str) -> list[float]:
        raw_action = self.engine.external_actions[agent_id]
        steering, throttle = self._compute_action(
            self.control_object,
            raw_action,
            self._decision_interval_s,
        )
        return [steering, throttle]

    def reset(self) -> None:
        super().reset()
        self.reset_controller_state()
```

Build the controllers from the validated config:

```python
def _build_controllers(self) -> None:
    speed = self._control_config.speed
    steering = self._control_config.steering
    self._speed_pid = BoundedPID(
        speed.kp, speed.ki, speed.kd, speed.integral_limit
    )
    self._heading_pid = BoundedPID(
        steering.heading_kp,
        steering.heading_ki,
        steering.heading_kd,
        steering.integral_limit,
    )
    self._lateral_pid = BoundedPID(
        steering.lateral_kp,
        steering.lateral_ki,
        steering.lateral_kd,
        steering.integral_limit,
    )


def reset_controller_state(self) -> None:
    self._speed_pid.reset()
    self._heading_pid.reset()
    self._lateral_pid.reset()
```

Keep the fail-safe wrapper separate from arithmetic:

```python
def _compute_action(
    self,
    vehicle: Any,
    action_value: DrivingAction | int,
    dt_s: float,
) -> tuple[float, float]:
    try:
        action = DrivingAction(action_value)
        steering, throttle, target = self._calculate_action(
            vehicle, action, dt_s
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        self.action_info = {
            "action": [0.0, -1.0],
            "fail_safe": True,
            "fail_safe_reason": type(exc).__name__,
        }
        return 0.0, -1.0
    self.action_info = {
        "action": [steering, throttle],
        "requested_action": int(action),
        "target_speed_mps": target,
        "steering": steering,
        "throttle_brake": throttle,
        "fail_safe": False,
        "fail_safe_reason": None,
    }
    return steering, throttle
```

Implement the calculation with the installed lane API and SI speeds:

```python
def _calculate_action(
    self,
    vehicle: Any,
    action: DrivingAction,
    dt_s: float,
) -> tuple[float, float, float]:
    navigation = getattr(vehicle, "navigation", None)
    lane = (
        getattr(navigation, "current_lane", None)
        if navigation is not None
        else None
    )
    if lane is None:
        lane = getattr(vehicle, "lane", None)
    if lane is None:
        raise ValueError("lane is unavailable")

    position = tuple(float(value) for value in vehicle.position)
    longitudinal, lateral = lane.local_coordinates(position)
    lookahead = self._control_config.steering.lookahead_m
    lane_heading = float(lane.heading_theta_at(longitudinal + lookahead))
    vehicle_heading = float(vehicle.heading_theta)
    heading_error = atan2(
        sin(lane_heading - vehicle_heading),
        cos(lane_heading - vehicle_heading),
    )
    heading_command = self._heading_pid.update(
        heading_error, dt_s, -1.0, 1.0
    )
    lateral_command = self._lateral_pid.update(
        float(lateral), dt_s, -1.0, 1.0
    )
    steering = min(max(heading_command + lateral_command, -1.0), 1.0)

    current_speed = float(vehicle.speed)
    speed_limit = self._speed_limit_mps(vehicle, lane)
    target = target_speed_mps(action, current_speed, speed_limit)
    speed_config = self._control_config.speed
    lower = (
        speed_config.emergency_deceleration_mps2
        if action is DrivingAction.STOP
        else speed_config.normal_deceleration_mps2
    )
    desired_acceleration = self._speed_pid.update(
        target - current_speed,
        dt_s,
        lower,
        speed_config.max_acceleration_mps2,
    )
    throttle = (
        desired_acceleration / speed_config.max_acceleration_mps2
        if desired_acceleration >= 0.0
        else desired_acceleration / abs(lower)
    )
    return steering, min(max(throttle, -1.0), 1.0), target


@staticmethod
def _speed_limit_mps(vehicle: Any, lane: Any) -> float:
    if hasattr(lane, "speed_limit"):
        return float(lane.speed_limit) / 3.6
    return float(vehicle.max_speed_m_s)
```

Import `atan2`, `cos`, and `sin` from `math`. The BoundedPID and target-speed validators turn non-finite runtime values into the fail-safe path.

- [ ] **Step 5: Bind the custom Policy through a MetaDrive subclass**

```python
def create_control_metadrive_env(
    config: dict[str, object],
    control_config: ControlConfig,
) -> DrivingEnvironment:
    from metadrive import MetaDriveEnv  # type: ignore[import-untyped]

    class ControlMetaDriveEnv(MetaDriveEnv):  # type: ignore[misc]
        @classmethod
        def default_config(cls) -> Any:
            defaults = super().default_config()
            defaults.update(
                {
                    "agent_policy": LaneKeepingLongitudinalPolicy,
                    "control_config": control_config.model_dump(),
                },
                allow_add_new_key=True,
            )
            return defaults

    return cast(DrivingEnvironment, ControlMetaDriveEnv(config))
```

If MetaDrive 0.4.3 requires `default_config()` to return a copied Config or rejects closure-based class values, document the observed error and move the same two values into a module-level subclass constructor without broadening scope.

- [ ] **Step 6: Add and run a real one-step integration test**

```python
@pytest.mark.integration
def test_control_policy_exposes_discrete_four_and_steps_headless() -> None:
    config = load_config("configs/base.yaml")
    env = create_control_metadrive_env(config.metadrive_dict(), config.control)
    try:
        env.reset(seed=config.seed)
        assert env.action_space.n == 4
        for action in DrivingAction:
            result = env.step(int(action))
            assert len(result) == 5
            assert math.isfinite(env.agent.steering)
            assert math.isfinite(env.agent.throttle_brake)
    finally:
        env.close()
```

Run: `.venv\Scripts\python.exe -m pytest tests\unit\control\test_lane_keeping_policy.py tests\integration\test_control_metadrive_headless.py -v`

Expected: unit and real MetaDrive Policy tests pass; only the documented upstream Matplotlib/Pyparsing warnings remain.

- [ ] **Step 7: Verify mypy and commit Policy boundary**

Run: `.venv\Scripts\python.exe -m mypy --strict src\mad_driving\control src\mad_driving\envs`

Expected: no issues in the new source files.

```powershell
git add src\mad_driving\control src\mad_driving\envs tests\unit\control tests\integration docs\phase3_implementation_log.md
git commit -m "feat: add lane keeping speed policy"
```

---

### Task 7: End-to-end control smoke and safe analysis fallback

**Files:**
- Modify: `src/mad_driving/envs/multi_agent_speed_env.py`
- Create: `src/mad_driving/cli/control_smoke.py`
- Create: `tests/unit/cli/test_control_smoke.py`

**Interfaces:**
- Produces: frozen `ControlSmokeResult` with final typed state, final trace, four action counts, and intervention count.
- Produces: `run_control_smoke(config, env_factory, suite_factory, coordinator_factory, shield_factory) -> ControlSmokeResult`.
- Produces CLI: `python -m mad_driving.cli.control_smoke --config configs/base.yaml`.

- [ ] **Step 1: Write failing lifecycle and action-count tests**

Use a fake environment that records integer actions and exposes the same finite vehicle boundary used by existing smoke tests. Inject small fake suite, Coordinator, and Shield implementations.

```python
def test_control_smoke_runs_decision_pipeline_and_closes() -> None:
    created: list[FakeControlEnv] = []

    def factory(options: dict[str, object], control: ControlConfig) -> FakeControlEnv:
        del control
        env = FakeControlEnv(options)
        created.append(env)
        return env

    result = run_control_smoke(make_config(decision_steps=4), env_factory=factory)
    env = created[0]
    assert env.actions == [0, 0]
    assert env.closed is True
    assert result.steps_completed == 2
    assert sum(result.action_counts) == result.steps_completed
    assert result.final_trace.executed_action == 0


def test_analysis_failure_executes_stop_and_still_closes() -> None:
    result, env = run_with_suite_that_raises(RuntimeError("analysis failed"))
    assert env.actions[0] == 3
    assert result.final_trace.executed_action == 3
    assert "multiple_agents_missing" in result.final_trace.shield_reasons
    assert result.final_review.reasons == ("agent_analysis_failed",)


def test_step_failure_always_closes() -> None:
    env = FakeControlEnv({}, fail_on_step=True)
    with pytest.raises(RuntimeError, match="step failed"):
        run_control_smoke(make_config(), env_factory=lambda options, control: env)
    assert env.closed is True
```

Also test monitor-mode counts no actual intervention, target speed in `DecisionTrace`, previous action/intervention propagation, termination and truncation, deterministic repeated fake runs, and finite CLI JSON.

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\cli\test_control_smoke.py -v`

Expected: import failure for `mad_driving.cli.control_smoke` or missing `ControlSmokeResult`.

- [ ] **Step 3: Implement typed result and fallback review**

Add:

```python
@dataclass(frozen=True)
class ControlSmokeResult:
    steps_completed: int
    terminated: bool
    truncated: bool
    final_snapshot: SceneSnapshot
    final_claims: tuple[RiskClaim, ...]
    final_review: CriticReview
    final_trace: DecisionTrace
    action_counts: tuple[int, int, int, int]
    shield_intervention_count: int
```

In `control_smoke.py`:

```python
def _fallback_analysis() -> tuple[tuple[RiskClaim, ...], CriticReview]:
    return (), CriticReview(
        conflict_score=1.0,
        unresolved_conflict=True,
        max_severity=1.0,
        supported_agent_ids=(),
        challenged_claim_ids=(),
        reasons=("agent_analysis_failed",),
    )


def _analyze_safely(
    suite: AnalysisSuite,
    snapshot: SceneSnapshot,
) -> tuple[tuple[RiskClaim, ...], CriticReview]:
    try:
        return suite.analyze(snapshot)
    except Exception:
        return _fallback_analysis()
```

The broad catch exists only at this deliberate fail-safe boundary. It must not wrap MetaDrive reset, step, snapshot construction, Coordinator, or Shield defects.

- [ ] **Step 4: Implement the exact pre-step decision loop**

Create the environment and all deterministic components once. Build step-0 snapshot and analysis after reset. For each decision:

```python
requested = coordinator.decide(snapshot, claims, review)
shield_result = shield.filter(requested, snapshot, claims)
executed = shield_result.executed_action
target = target_speed_mps(
    executed,
    snapshot.ego.speed_mps,
    snapshot.ego.speed_limit_mps,
)
_, _, terminated, truncated, _ = env.step(int(executed))
trace = DecisionTrace(
    step_index=step_index,
    raw_action=int(requested),
    executed_action=int(executed),
    target_speed_mps=target,
    shield_intervened=shield_result.intervened,
    shield_reasons=shield_result.reasons,
    claims=claims,
    review=review,
    reward_components={},
)
snapshot = builder.build(
    env,
    step_index=step_index,
    scenario_id=config.scenario_id,
    seed=config.seed,
    previous_action=int(executed),
    previous_shield_intervention=shield_result.intervened,
)
claims, review = _analyze_safely(suite, snapshot)
```

Increment the selected action count and actual intervention count exactly once per completed `env.step`. Keep environment closure in `finally`. Refuse to return if no decision step completed.

- [ ] **Step 5: Implement finite JSON CLI output**

Reuse the Phase 2 CLI pattern: strict `--config`, traceback-free operational error with exit code 2, and `json.dumps(asdict(result), ensure_ascii=False, sort_keys=True)`. Unit-test JSON using `parse_constant` that raises on NaN/Infinity.

- [ ] **Step 6: Verify GREEN and fixed-smoke regression**

Run: `.venv\Scripts\python.exe -m pytest tests\unit\cli\test_control_smoke.py tests\unit\cli\test_smoke.py -v`

Expected: control lifecycle, fallback STOP, closure, action counts, trace, finite JSON, and unchanged fixed-action smoke all pass.

- [ ] **Step 7: Commit control smoke independently**

```powershell
git add src\mad_driving\envs\multi_agent_speed_env.py src\mad_driving\cli\control_smoke.py tests\unit\cli
git commit -m "feat: run shielded rule-based control smoke"
```

---

### Task 8: Real 100-step control verification, documentation, and publish

**Files:**
- Modify: `tests/integration/test_control_metadrive_headless.py`
- Modify: `README.md`
- Modify: `docs/phase3_implementation_log.md`
- Modify: `docs/superpowers/plans/2026-07-20-phase3-control-shield.md`

**Interfaces:**
- Verifies real MetaDrive lifecycle, Discrete(4), finite low-level commands, lane keeping, forced STOP deceleration, Agent decisions, Shield results, typed trace, and closure.
- Documents exact dependency versions, API observations, deviations, test count, branch coverage, warnings, and simulated seconds.

- [ ] **Step 1: Add a real forced-STOP integration assertion**

Create a short test that accelerates from reset with KEEP, captures speed, then applies STOP for enough decisions to observe deceleration:

```python
@pytest.mark.integration
def test_real_policy_stop_reduces_speed() -> None:
    config = load_config("configs/base.yaml")
    env = create_control_metadrive_env(config.metadrive_dict(), config.control)
    try:
        env.reset(seed=config.seed)
        for _ in range(20):
            env.step(int(DrivingAction.KEEP))
        speed_before_stop = env.agent.speed
        for _ in range(20):
            env.step(int(DrivingAction.STOP))
        assert speed_before_stop > 0.0
        assert env.agent.speed < speed_before_stop
    finally:
        env.close()
```

If MetaDrive terminates before either loop completes, break on terminated/truncated and assert only when the required samples exist; record this installed-API behavior. Do not disable collision or out-of-road termination merely to force the assertion.

- [ ] **Step 2: Run targeted real integration and control smoke**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\integration\test_control_metadrive_headless.py -v
.venv\Scripts\uv.exe sync --no-editable --group dev --reinstall-package mad-driving
.venv\Scripts\python.exe -m mad_driving.cli.control_smoke --config configs\base.yaml
```

Expected:

- integration exits zero;
- control smoke completes 100 decisions unless MetaDrive reports a legitimate termination/truncation;
- `sum(action_counts) == steps_completed`;
- final JSON contains finite snapshot, claims, review, trace, and intervention count;
- no render window opens.

- [ ] **Step 3: Prove the old fixed-action smoke still works**

Run: `.venv\Scripts\python.exe -m mad_driving.cli.smoke --config configs\base.yaml`

Expected: 100-step Phase 2 fixed-action JSON remains present and does not use Coordinator, Shield, or custom Policy.

- [ ] **Step 4: Update README and Phase 3 evidence**

README must state:

- Phase 3 scope and module boundaries;
- fixed smoke versus control smoke commands;
- four Action names and Safety Shield modes;
- control smoke output fields;
- Phase 4 exclusions.

`docs/phase3_implementation_log.md` must record:

- Python, MetaDrive, Gymnasium, Pydantic, pytest, Ruff, mypy versions;
- exact custom Policy API observed in installed MetaDrive 0.4.3;
- every mismatch and minimal alternative;
- test count and branch coverage;
- static-check results;
- control and fixed smoke step counts and simulated seconds;
- action counts and intervention count from the canonical control smoke;
- all accepted upstream warnings.

- [ ] **Step 5: Run the complete quality gate fresh**

```powershell
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
git diff --check feat/phase2-deterministic-agents...HEAD
git diff --check
```

Expected: all tests pass, total branch coverage is at least 80%, Ruff and formatting report no changes, mypy reports no issues, and both diff checks exit zero. Only documented upstream Matplotlib/Pyparsing warnings are accepted.

- [ ] **Step 6: Review scope and commit verification evidence**

Run:

```powershell
rg -n "ObservationBuilder|reward_function|PPO|stable_baselines|ScenarioManager|T[O]DO|FIX[M]E|NotImpl[e]mented" src tests configs
rg -n "env\.step|fixed_action|executed_action|steering" src\mad_driving\cli src\mad_driving\coordinator src\mad_driving\safety
git status --short
```

Expected: no Phase 4 implementation or placeholder appears; `control_smoke` passes only the Shield-executed integer Action; Coordinator and Shield contain no steering logic; unrelated files are absent from status.

```powershell
git add README.md docs src tests configs
git commit -m "test: verify phase 3 shielded control"
```

- [ ] **Step 7: Push and open a stacked Draft PR**

After confirming a clean worktree and the fresh quality evidence:

```powershell
git push -u origin feat/phase3-control-shield
```

Open a Draft PR with:

- base: `feat/phase2-deterministic-agents`
- head: `feat/phase3-control-shield`
- title: `Phase 3: add shielded lane and speed control`
- body: architecture, four actions, Shield guarantees, fail-safe behavior, test/coverage evidence, real 100-step simulation evidence, and a note to retarget to `main` after PR #1 and PR #2 merge.

---

## Self-review record

- **Spec coverage:** Tasks 2–7 implement all approved Phase 3 units: four-action mapping, rule Coordinator, monotone three-mode Shield, bounded PID with anti-windup, lane keeping custom Policy, and end-to-end control smoke. Task 8 covers real MetaDrive and acceptance evidence.
- **Scope:** The plan does not create the 24-dimensional Observation, Reward, Gymnasium wrapper, PPO, ScenarioManager, persistent traces, or evaluation artifacts.
- **Safety:** Shield validation occurs before arithmetic, enforce mode uses only `max(requested, required)`, Agent analysis failure becomes missing claims and STOP, and Policy failure becomes `[0.0, -1.0]`.
- **Type consistency:** `DrivingAction` is shared by ActionMapper, Coordinator, ShieldResult, Shield, Policy, smoke, and tests. `AppConfig.control` feeds the runtime MetaDrive subclass; the Python Policy class is not serialized into YAML.
- **Lifecycle consistency:** Decisions use pre-step claims; traces retain those claims; next snapshots carry the executed action and actual intervention flag; counts increment once per completed step.
- **Verification:** Every production task starts with an explicit failing test, names the expected RED symptom, runs targeted GREEN checks, and ends with an independent commit.
