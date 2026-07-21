# Phase 1 Foundation Implementation Plan

> **Status: historical Phase 1 plan.** This file preserves the implementation record from the repository foundation. The active system contract is `docs/multi_agent_driving_mvp_spec.md`; Phase 4.1 hardening is specified in `docs/superpowers/specs/2026-07-21-phase4-research-validity-hardening-design.md` and tracked in `docs/superpowers/plans/2026-07-21-phase4-research-validity-hardening.md`.

Phase 4.1 now implements the research-hardening contract while retaining the 24-dimensional Observation:

- three specialists plus one Critic and one-to-three claims per specialist, conservatively aggregated into 24 slots
- standard PPO rollout updates and timing `0.02 / 5 / 0.10`
- role-disjoint train/validation/test seeds, actual-reset JSONL artifacts, and a documented five-run policy/RNG-seed comparison protocol
- `ScenarioRuntime`, hidden-kinematics isolation, boundary-causal Reward, and coordinate signs
- all-monitor decision comparison, all-enforce system comparison, and fresh destination/provenance version 3

The explicit validity features `ttc_valid`, `claim_valid`, `agent_failed`, and `target_actor_present` remain deferred and unimplemented.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reproducible Phase 1 foundation: strict configuration, validated JSON-serializable interfaces, MetaDrive-backed `SceneSnapshot` creation, and a deterministic headless fixed-action smoke run.

**Architecture:** Keep simulator integration at the boundary. Immutable interface dataclasses and strict configuration are simulator-independent; `SceneSnapshotBuilder` projects MetaDrive runtime state into those interfaces; the smoke CLI owns the environment lifecycle and guarantees `close()` through a context manager. Phase 1 does not implement agents, the 24-dimensional coordinator observation, PPO, scenarios, reward shaping, or the Safety Shield.

**Tech Stack:** Python 3.11, MetaDrive 0.4.3, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, PyTorch 2.8+, NumPy, Pydantic v2, PyYAML, pytest, Ruff, mypy, uv.

## Global Constraints

- Treat `docs/multi_agent_driving_mvp_spec.md` as the highest-level requirement.
- Use Python 3.11; MetaDrive 0.4.3 explicitly requires Python `>=3.6,<3.12`.
- Use Gymnasium's `reset -> (observation, info)` and `step -> (observation, reward, terminated, truncated, info)` API.
- Keep RGB observations, LLM/VLM calls, learned steering, route planning, multi-ego learning, and online weight updates out of Phase 1.
- Headless execution is the default and must work on CPU.
- All data models are immutable, fully type-annotated, finite-valued, and serializable through `dataclasses.asdict` plus `json.dumps`.
- Unknown configuration keys and invalid ranges must stop startup.
- Use TDD: each production behavior requires an observed failing test before implementation.
- Run Windows paths through `pathlib.Path`; do not hard-code separators.
- Pin direct dependencies in `pyproject.toml` and commit `uv.lock` after the verified environment is resolved.

---

### Task 1: Reproducible project skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/mad_driving/__init__.py`
- Create: package `__init__.py` files under `interfaces/`, `config/`, `world_model/`, `envs/`, and `cli/`
- Create: `configs/base.yaml`

**Interfaces:**
- Produces: installable package `mad_driving`
- Produces: console-independent module entry point `python -m mad_driving.cli.smoke`
- Produces: a locked Python 3.11 environment

- [x] **Step 1: Add packaging and test configuration**

Use a `src` layout, require Python `>=3.11,<3.12`, and configure pytest to use `tests/`. Pin MetaDrive/Gymnasium/SB3; keep tooling in a `dev` dependency group.

- [x] **Step 2: Add the minimum base YAML**

```yaml
seed: 42
scenario_id: phase1_smoke
decision_steps: 100
fixed_action: [0.0, 0.25]
metadrive:
  use_render: false
  image_observation: false
  num_scenarios: 1
  start_seed: 42
  traffic_density: 0.1
  horizon: 200
```

- [x] **Step 3: Resolve and lock dependencies**

Run: `python -m venv .venv`, install `uv` inside `.venv`, then run `.venv/Scripts/uv.exe lock` and `.venv/Scripts/uv.exe sync --all-groups` on Windows.

Expected: `uv.lock` exists; `python`, `metadrive`, `gymnasium`, and test tools import under `.venv`.

- [x] **Step 4: Verify package imports before behavior exists**

Run: `.venv/Scripts/python.exe -c "import mad_driving; print(mad_driving.__version__)"`

Expected: prints the package version and exits 0.

---

### Task 2: Strict configuration loader

**Files:**
- Create: `tests/unit/config/test_loader.py`
- Create: `src/mad_driving/config/models.py`
- Create: `src/mad_driving/config/loader.py`

**Interfaces:**
- Produces: `AppConfig`
- Produces: `load_config(path: str | Path) -> AppConfig`
- `AppConfig.metadrive_dict() -> dict[str, object]` returns only MetaDrive-supported configuration keys.

- [x] **Step 1: Write failing tests**

Tests must prove that the base YAML loads, missing files fail clearly, unknown root/nested keys are rejected, `decision_steps < 1` is rejected, malformed two-element actions are rejected, and non-finite action values are rejected.

```python
def test_unknown_root_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\nscenario_id: x\ndecision_steps: 1\nfixed_action: [0, 0]\nmetadrive: {}\nextra: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)
```

- [x] **Step 2: Run tests and verify RED**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/config/test_loader.py -v`

Expected: collection/import failure because `mad_driving.config.loader` does not exist.

- [x] **Step 3: Implement strict Pydantic models and YAML loading**

Use `ConfigDict(extra="forbid", frozen=True)`, finite floats, and `Path.read_text(encoding="utf-8")`. Raise `FileNotFoundError` for absent paths and preserve Pydantic's field-specific validation errors.

- [x] **Step 4: Run tests and verify GREEN**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/config/test_loader.py -v`

Expected: all config tests pass without warnings.

---

### Task 3: Immutable validated interfaces

**Files:**
- Create: `tests/unit/interfaces/test_models.py`
- Create: `src/mad_driving/interfaces/actor_state.py`
- Create: `src/mad_driving/interfaces/scene_snapshot.py`
- Create: `src/mad_driving/interfaces/risk_claim.py`
- Create: `src/mad_driving/interfaces/critic_review.py`
- Create: `src/mad_driving/interfaces/decision_trace.py`
- Create: `src/mad_driving/interfaces/_validation.py`

**Interfaces:**
- Produces the exact dataclasses from specification section 5: `ActorState`, `EgoState`, `SceneSnapshot`, `RiskClaim`, `CriticReview`, `DecisionTrace`.
- `_validation.require_finite(name: str, value: float) -> None`
- `_validation.require_probability(name: str, value: float) -> None`

- [x] **Step 1: Write failing model tests**

Cover construction, immutability, `asdict`/JSON serialization, invalid actor types, negative sizes/speeds, invalid probabilities/confidence/severity, negative recommended speed, invalid action indices, and NaN/infinity in every float-bearing model.

```python
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_risk_claim_rejects_non_finite_confidence(bad: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        make_claim(confidence=bad)
```

- [x] **Step 2: Run tests and verify RED**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/interfaces/test_models.py -v`

Expected: collection/import failure because interface modules do not exist.

- [x] **Step 3: Implement the minimal frozen dataclasses**

Validation belongs in `__post_init__`; use tuple values and copied dictionaries so callers cannot mutate a frozen model through retained mutable objects.

- [x] **Step 4: Run tests and verify GREEN**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/interfaces/test_models.py -v`

Expected: all interface tests pass.

---

### Task 4: MetaDrive `SceneSnapshotBuilder`

**Files:**
- Create: `tests/unit/world_model/test_snapshot_builder.py`
- Create: `src/mad_driving/world_model/validation.py`
- Create: `src/mad_driving/world_model/snapshot_builder.py`

**Interfaces:**
- Produces: `SceneSnapshotBuilder.build(env: MetaDriveEnv, *, step_index: int, scenario_id: str, seed: int, previous_action: int, previous_shield_intervention: bool) -> SceneSnapshot`
- Consumes the installed MetaDrive 0.4.3 vehicle state API after direct source inspection.

- [x] **Step 1: Inspect installed MetaDrive source**

Record the concrete 0.4.3 attributes used for ego position, heading, speed, navigation lane, route completion, and engine objects in this plan's implementation log before coding.

- [x] **Step 2: Write failing tests with small protocol-conforming fakes**

Tests cover SI-unit conversion, actor-relative coordinates, stable actor ordering, missing optional navigation information, invalid/non-finite simulator state, and identical output for identical state.

- [x] **Step 3: Run tests and verify RED**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/world_model/test_snapshot_builder.py -v`

Expected: import failure because `SceneSnapshotBuilder` does not exist.

- [x] **Step 4: Implement the minimal adapter**

Use only documented/inspected MetaDrive 0.4.3 attributes. Convert `speed_km_h` to m/s, compute relative coordinates in the ego frame, and sort actors by `actor_id` before constructing the tuple.

- [x] **Step 5: Run tests and verify GREEN**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/world_model/test_snapshot_builder.py -v`

Expected: all snapshot tests pass.

---

### Task 5: Guaranteed lifecycle and fixed-action smoke CLI

**Files:**
- Create: `tests/unit/cli/test_smoke.py`
- Create: `src/mad_driving/envs/multi_agent_speed_env.py`
- Create: `src/mad_driving/cli/smoke.py`

**Interfaces:**
- Produces: `run_smoke(config: AppConfig, env_factory: Callable[[dict[str, object]], MetaDriveEnv] = MetaDriveEnv) -> SmokeResult`
- Produces: immutable `SmokeResult(steps_completed, terminated, truncated, final_snapshot)`
- CLI: `python -m mad_driving.cli.smoke --config configs/base.yaml`

- [x] **Step 1: Write failing tests using a real in-memory fake environment**

Prove that `reset(seed=config.seed)` is called once, the configured fixed action is applied, termination/truncation stops the loop, snapshots are built, and `close()` runs both on success and when `step()` raises.

- [x] **Step 2: Run tests and verify RED**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/cli/test_smoke.py -v`

Expected: import failure because `run_smoke` does not exist.

- [x] **Step 3: Implement lifecycle and CLI**

Use `try/finally` around the full reset/step lifecycle. Print one JSON result object to stdout; return nonzero with a clear stderr message for invalid config or simulator failure.

- [x] **Step 4: Run tests and verify GREEN**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/cli/test_smoke.py -v`

Expected: all smoke CLI unit tests pass.

---

### Task 6: Real MetaDrive headless integration

**Files:**
- Create: `tests/integration/test_metadrive_headless.py`
- Modify: `configs/base.yaml`
- Modify: `docs/implementation_plan.md`

**Interfaces:**
- Verifies MetaDrive 0.4.3 `reset/step/close` behavior under the locked environment.

- [x] **Step 1: Write the integration test**

The test creates the real environment with `use_render=false`, resets with seed 42, performs at least one fixed action, verifies Gymnasium's five-element step result, builds a finite snapshot, and closes in `finally`.

- [x] **Step 2: Run and verify RED before completing integration support**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_metadrive_headless.py -v`

Expected: fail on the first concrete mismatch between the unit-test adapter assumptions and installed MetaDrive, or fail because integration wiring is incomplete.

- [x] **Step 3: Apply the smallest API-compatible correction**

If the installed API differs from the specification assumptions, document the mismatch, cause, and minimal alternative in `docs/implementation_plan.md`; do not broaden scope.

- [x] **Step 4: Run the real CLI**

Run: `.venv/Scripts/python.exe -m mad_driving.cli.smoke --config configs/base.yaml`

Expected: exit 0, exactly the configured number of steps unless terminated/truncated earlier, finite final snapshot, and no visible window.

---

### Task 7: Phase 1 quality gate and handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/implementation_plan.md`

**Interfaces:**
- Produces exact setup and smoke commands for Windows and Linux.
- Produces a Phase 1 implementation log with dependency versions and verification evidence.

- [x] **Step 1: Run targeted and full verification**

```powershell
.venv\Scripts\python.exe -m pytest -v
.venv\Scripts\ruff.exe check .
.venv\Scripts\mypy.exe src
.venv\Scripts\python.exe -m mad_driving.cli.smoke --config configs/base.yaml
```

Expected: every command exits 0. The test report contains unit plus real MetaDrive integration coverage.

- [x] **Step 2: Review the diff and specification coverage**

Run: `git diff --check` and `git diff --stat`.

Confirm Phase 1 includes only foundation work and no Phase 2+ behavior.

- [x] **Step 3: Commit Phase 1**

```powershell
git add pyproject.toml uv.lock .gitignore README.md configs docs src tests
git commit -m "feat: implement phase 1 MetaDrive foundation"
git push origin main
```

Expected: one independent Phase 1 commit is present on `kazu02210679/drive`.

---

## Subsequent phase boundaries

- Phase 2 receives a separate TDD plan for Nominal, Hazard, Rule, Critic, and DecisionTrace logging.
- Phase 3 receives a separate TDD plan for action mapping, PID, lane keeping, Safety Shield, and a rule coordinator.
- Phase 4 receives a separate TDD plan for the 24-dimensional observation, reward, Gymnasium wrapper checking, PPO, checkpoints, and TensorBoard.
- Phase 5 receives a separate TDD plan for the three seeded scenarios and curriculum.
- Phase 6 receives a separate TDD plan for baselines, ablations, metrics, plots, GIF overlay, and Markdown reporting.

## Self-review record

- Spec coverage: every Phase 1 item in sections 5, 10, 18, 19, 21, 22, and 25 maps to Tasks 1-7.
- Deliberate exclusions: Agents, Safety Shield, the final 24-dimensional observation, reward, PPO, scenario actors, comparison experiments, and visualization remain in their specified later phases.
- Placeholder scan: all Phase 1 steps contain concrete files, commands, behavior, and expected evidence; later phases are explicit plan boundaries.
- Type consistency: `AppConfig`, `SceneSnapshotBuilder.build`, `SmokeResult`, and `run_smoke` signatures are consistent across producer and consumer tasks.

## Implementation log

### Windows Unicode path and editable installation

- **Problem:** On this Windows host, Hatch's editable installation wrote the project path into `_editable_impl_mad_driving.pth` with a mojibake user-directory segment. `pip show` reported the project as installed, but `import mad_driving` failed because the `.pth` target did not exist.
- **Cause:** The failure is in editable-path serialization for the non-ASCII workspace path; the package source and MetaDrive installation are intact.
- **Minimal alternative:** Use `uv sync --no-editable --group dev` for the verified Windows environment. This installs a normal wheel into the virtual environment and avoids the corrupted path indirection. Re-run sync after package source changes when a direct installed-package verification is required; pytest and static checks still run against repository source.
- **Scope impact:** No feature or repository-layout change. Linux and ASCII-only Windows paths may continue to use editable installs, but documented reproducible commands use `--no-editable`.

### MetaDrive 0.4.3 speed and acceleration projection

- **Observed API:** `BaseObject.speed` and `BaseObject.velocity` are already m/s, while `lane.speed_limit` is compared with `speed_km_h` and is therefore km/h. `BaseVehicle.before_step()` stores `last_velocity`; the decision interval is `physics_world_step_size * decision_repeat`.
- **Minimal correction to Task 4 wording:** Build ego speed directly from `vehicle.speed`, convert only `lane.speed_limit / 3.6`, and derive acceleration from current minus `last_velocity` over the configured decision interval. This avoids a redundant round-trip through km/h and uses inspected 0.4.3 attributes.

### MetaDrive 0.4.3 Config compatibility

- **Observed API:** A live `MetaDriveEnv.config` is `metadrive.utils.config.Config`, which exposes `get(key, default)` but does not implement `collections.abc.Mapping`.
- **Minimal correction:** The snapshot boundary now accepts the structural `get(key, default)` interface shared by `dict` and MetaDrive `Config`. Strict startup validation remains in `AppConfig`; no feature or configuration schema changed.

### MetaDrive ego accessor deprecation

- **Observed API:** MetaDrive 0.4.3 still provides `env.vehicle`, but emits a deprecation warning and directs integrations to `env.agent`.
- **Minimal correction:** Prefer `env.agent` and retain `env.vehicle` only as a fake/older-runtime fallback. The snapshot schema and behavior are unchanged.

### Phase 1 verification evidence

- **Runtime:** Python 3.11.9, MetaDrive 0.4.3, Gymnasium 1.3.0.
- **Automated tests:** 62 passed, including one real headless MetaDrive integration test; branch coverage 92.86% (required minimum: 80%).
- **Static checks:** Ruff lint and format checks passed; mypy strict passed for 18 source files.
- **Smoke run:** `configs/base.yaml` completed 100 decision steps / 10.0 simulated seconds, exited 0, opened no visible window, and emitted a finite final `SceneSnapshot` as JSON.
- **Known warnings:** MetaDrive's transitive Matplotlib stack emits 14 upstream pyparsing deprecation warnings during the integration test. They do not affect execution or project code checks.
