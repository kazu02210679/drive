# Phase 5 Scenarios and Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three reproducible MetaDrive hazard scenarios and a validation-driven Levels 0-3 curriculum without changing the 24-dimensional Coordinator observation.

**Architecture:** Pure Python `ScenarioRuntime` implementations own seeded parameters, behavior, visibility, and outcomes. A thin MetaDrive `ScenarioActorManager` owns simulator objects and physics lifecycle; a training callback owns curriculum progression and broadcasts pending levels at reset boundaries.

**Tech Stack:** Python 3.11, MetaDrive 0.4.3, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, NumPy 1.26.4, Pydantic 2.11.7, PyYAML 6.0.2, pytest 8.4.1.

## Global Constraints

- Treat `docs/multi_agent_driving_mvp_spec.md` and `docs/superpowers/specs/2026-07-22-phase5-scenarios-design.md` as the requirements hierarchy.
- Preserve `OBSERVATION_SCHEMA_VERSION = 1`, observation shape `(24,)`, and action order `KEEP`, `SLOW`, `PREPARE_STOP`, `STOP`.
- Use only `scenario_parameter_seed` for scenario selection and sampled Actor parameters.
- Keep train `[0, 10000)`, validation `[10000, 11000)`, and test `[20000, 21000)` disjoint.
- Never use test episodes for training, checkpoint selection, or curriculum progression.
- Agent-visible state must exclude hidden crossing Actors; privileged truth must retain them with `visible=False, occluded=True`.
- Agent Claims must not be used by Reward or scenario outcome calculations.
- Simulator integration errors fail fast; physical collision outcomes remain normal episode outcomes.
- Every implementation change follows red-green-refactor TDD and ends in one of the four Phase 5 review commits.
- Final quality gates are Ruff, strict mypy, complete pytest with at least 90% coverage, real headless MetaDrive smokes, and identical-seed repeatability.

## File map

### Common infrastructure and Lead Brake

- Modify `src/mad_driving/config/models.py`: strict scenario and curriculum Pydantic models.
- Modify `src/mad_driving/config/loader.py`: ordered recursive YAML overlay loading.
- Modify `configs/base.yaml`: validated Phase 5 defaults while retaining Phase 4 smoke behavior.
- Create `configs/scenarios/lead_brake.yaml`: fixed Lead Brake overlay.
- Create `src/mad_driving/scenarios/actor_adapter.py`: simulator-neutral geometry, spawn, command, and state contracts.
- Create `src/mad_driving/scenarios/actor_manager.py`: MetaDrive object ownership and command application.
- Create `src/mad_driving/scenarios/parameters.py`: local seeded sampling helpers.
- Create `src/mad_driving/scenarios/manager.py`: level selection, runtime delegation, pending-level boundary.
- Create `src/mad_driving/scenarios/nominal.py`: meaningful Level 0 outcome runtime.
- Create `src/mad_driving/scenarios/lead_brake.py`: Lead Brake runtime.
- Modify `src/mad_driving/envs/control_metadrive_env.py`: register and expose the Actor manager.
- Modify `src/mad_driving/envs/multi_agent_speed_env.py`: use the Phase 5 runtime factory and return typed scenario info.
- Modify `src/mad_driving/scenarios/__init__.py`: public exports.
- Create `tests/unit/scenarios/fakes.py`: reusable fake Actor adapter.
- Create `tests/unit/scenarios/test_parameters.py`.
- Create `tests/unit/scenarios/test_manager.py`.
- Create `tests/unit/scenarios/test_lead_brake.py`.
- Create `tests/unit/scenarios/test_actor_manager.py`.
- Modify `tests/unit/config/test_loader.py` and `tests/unit/config/test_rl_config.py`.
- Create `tests/integration/test_phase5_metadrive_headless.py`.

### Cut-in

- Create `configs/scenarios/cut_in.yaml`.
- Create `src/mad_driving/scenarios/cut_in.py`.
- Create `tests/unit/scenarios/test_cut_in.py`.
- Extend `tests/integration/test_phase5_metadrive_headless.py`.

### Occluded Crossing Actor

- Create `configs/scenarios/occluded_crossing.yaml`.
- Create `src/mad_driving/scenarios/occluded_crossing.py`.
- Modify `src/mad_driving/world_model/snapshot_builder.py` only if the real Cyclist adapter exposes an incompatible stable Actor ID; do not change visibility semantics.
- Create `tests/unit/scenarios/test_occluded_crossing.py`.
- Extend `tests/integration/test_phase5_metadrive_headless.py`.

### Curriculum, provenance, and completion

- Create `src/mad_driving/training/curriculum.py`: state machine and immutable snapshot.
- Modify `src/mad_driving/training/callbacks.py`: validation metric capture and level broadcast.
- Modify `src/mad_driving/training/train.py`: runtime factory, curriculum callback, atomic state artifact, and resume wiring.
- Modify `src/mad_driving/training/episode_seeds.py`: schema v3 scenario-generation records.
- Modify `src/mad_driving/training/metadata.py`: research contract v5 and curriculum provenance.
- Modify `src/mad_driving/interfaces/decision_trace.py`: concrete scenario ID and difficulty level.
- Modify `README.md`: Phase 5 configuration and smoke commands.
- Create `tests/unit/training/test_curriculum.py`.
- Modify `tests/unit/training/test_callbacks.py`, `test_episode_seeds.py`, `test_metadata.py`, and `test_train.py`.
- Modify `tests/unit/interfaces/test_models.py` and `tests/unit/envs/test_multi_agent_speed_env.py`.
- Extend `tests/integration/test_rl_metadrive_headless.py` and `test_ppo_checkpoint.py`.

---

### Task 1: Common infrastructure and Lead Brake

**Files:**
- Create/modify every file listed under “Common infrastructure and Lead Brake”.

**Interfaces:**
- Produces `build_scenario_runtime_factory(config: AppConfig) -> ScenarioRuntimeFactory`.
- Produces `MultiAgentSpeedEnv.set_difficulty_level(level: int) -> None`, applied on the next reset only.
- Produces immutable `RoadGeometry`, `LaneVehicleSpawn`, `KinematicActorSpawn`, `StaticOccluderSpawn`, `ActorCommand`, and `ScenarioActorState` dataclasses.
- Produces adapter methods `scenario_road_geometry()`, `scenario_spawn_lane_vehicle()`, `scenario_spawn_crossing_actor()`, `scenario_spawn_occluder()`, `scenario_command_actor()`, `scenario_actor_state()`, and `scenario_actor_ids()`.
- Produces `ScenarioManagerRuntime` and concrete scenario ID `lead_brake`.

- [ ] **Step 1: Write failing strict-config and overlay tests**

Add tests that prove default values, range ordering, unknown-key rejection, recursive merge, and conflict rejection:

```python
def test_phase5_defaults_are_strict() -> None:
    config = load_config("configs/base.yaml")
    assert config.scenarios.curriculum.mode == "fixed"
    assert config.scenarios.curriculum.fixed_level == 0
    assert config.scenarios.lead_brake.initial_gap_m.minimum == 35.0
    assert config.scenarios.lead_brake.initial_gap_m.maximum == 55.0


def test_overlay_selects_fixed_lead_brake(tmp_path: Path) -> None:
    config = load_config("configs/base.yaml", "configs/scenarios/lead_brake.yaml")
    assert config.scenarios.curriculum.fixed_level == 1
    assert config.scenarios.selection == "lead_brake"


def test_overlay_rejects_mapping_scalar_conflict(tmp_path: Path) -> None:
    overlay = tmp_path / "bad.yaml"
    overlay.write_text("scenarios:\n  lead_brake: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping conflict"):
        load_config("configs/base.yaml", overlay)
```

- [ ] **Step 2: Run the config tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/config/test_loader.py tests/unit/config/test_rl_config.py -q`

Expected: FAIL because `curriculum`, `lead_brake`, `selection`, and overlay arguments do not exist.

- [ ] **Step 3: Implement strict config models and recursive overlays**

Add these core model boundaries to `config/models.py` and nest them in the existing scenarios model:

```python
class FloatRangeConfig(StrictTypedFrozenModel):
    minimum: FiniteFloat
    maximum: FiniteFloat

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class CurriculumConfig(StrictTypedFrozenModel):
    mode: Literal["fixed", "automatic"] = "fixed"
    fixed_level: int = Field(default=0, ge=0, le=3)
    initial_level: int = Field(default=0, ge=0, le=3)
    success_rate_threshold: FiniteFloat = Field(default=0.80, ge=0.0, le=1.0)
    collision_rate_threshold: FiniteFloat = Field(default=0.05, ge=0.0, le=1.0)
    consecutive_evaluations: PositiveInt = 2


class LeadBrakeScenarioConfig(StrictTypedFrozenModel):
    initial_gap_m: FloatRangeConfig = FloatRangeConfig(minimum=35.0, maximum=55.0)
    speed_fraction: FloatRangeConfig = FloatRangeConfig(minimum=0.80, maximum=1.00)
    trigger_s: FloatRangeConfig = FloatRangeConfig(minimum=1.0, maximum=3.0)
    mild_deceleration_mps2: FloatRangeConfig = FloatRangeConfig(minimum=2.0, maximum=4.0)
    severe_deceleration_mps2: FloatRangeConfig = FloatRangeConfig(minimum=4.0, maximum=8.0)
    survival_s: FiniteFloat = Field(default=4.0, gt=0.0)
```

Change the loader signature and use a recursive copy that rejects mapping/scalar replacement:

```python
def load_config(path: str | Path, *overlays: str | Path) -> AppConfig:
    payload = _load_yaml_mapping(Path(path))
    for overlay in overlays:
        payload = _merge_mapping(payload, _load_yaml_mapping(Path(overlay)), path="")
    return AppConfig.model_validate(payload)
```

Keep `configs/base.yaml` in fixed Level 0 mode so Phase 4 tests remain nominal. The Lead Brake overlay sets `scenario_id: phase5`, `selection: lead_brake`, and fixed Level 1.

- [ ] **Step 4: Run config tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/config -q`

Expected: PASS.

- [ ] **Step 5: Write failing deterministic parameter and manager-selection tests**

Create tests with exact seed behavior and reset-boundary level application:

```python
def test_sampler_repeats_identical_values() -> None:
    first = ScenarioParameterSampler(123).uniform("gap", 35.0, 55.0)
    second = ScenarioParameterSampler(123).uniform("gap", 35.0, 55.0)
    assert first == second


def test_manager_uses_only_parameter_seed() -> None:
    first = runtime_state(EpisodeSeeds(1, 7, 99), level=2)
    second = runtime_state(EpisodeSeeds(500, 800, 99), level=2)
    assert first.scenario_id == second.scenario_id
    assert first.parameters == second.parameters


def test_pending_level_applies_only_on_next_reset() -> None:
    runtime = make_manager(level=0)
    state = runtime.reset(fake_environment(), seeds=EpisodeSeeds(1, 2, 3))
    runtime.set_difficulty_level(2)
    assert state.parameters["difficulty_level"] == 0
    next_state = runtime.reset(fake_environment(), seeds=EpisodeSeeds(4, 5, 6))
    assert next_state.parameters["difficulty_level"] == 2
```

- [ ] **Step 6: Run selection tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_parameters.py tests/unit/scenarios/test_manager.py -q`

Expected: FAIL because the sampler and manager runtime do not exist.

- [ ] **Step 7: Implement deterministic sampling and manager delegation**

Use one local generator and stable scenario tables:

```python
_SCENARIOS_BY_LEVEL: dict[int, tuple[str, ...]] = {
    0: ("nominal",),
    1: ("lead_brake",),
    2: ("lead_brake", "cut_in"),
    3: ("occluded_crossing",),
}


class ScenarioParameterSampler:
    def __init__(self, seed: int) -> None:
        self._generator = np.random.default_rng(seed)

    def uniform(self, name: str, minimum: float, maximum: float) -> float:
        del name
        return float(self._generator.uniform(minimum, maximum))

    def choose(self, values: tuple[str, ...]) -> str:
        return values[int(self._generator.integers(0, len(values)))]
```

`ScenarioManagerRuntime.reset()` must select first, then create the concrete runtime, then sample its parameters with the same generator. Store `difficulty_level` and every generated value in `ScenarioState.parameters`.

- [ ] **Step 8: Run selection tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_parameters.py tests/unit/scenarios/test_manager.py -q`

Expected: PASS.

- [ ] **Step 9: Write failing Actor adapter/manager ownership tests**

Test registration, exact manager forwarding, duplicate IDs, command-before-spawn rejection, and cleanup:

```python
def test_actor_manager_owns_and_clears_spawned_actor() -> None:
    manager = manager_with_fake_engine()
    actor_id = manager.spawn_lane_vehicle(
        LaneVehicleSpawn("lead", (">", ">>", 0), 40.0, 0.0, 8.0)
    )
    assert actor_id == "lead"
    manager.before_reset()
    assert manager.actor_ids() == ()


def test_command_rejects_unknown_actor() -> None:
    manager = manager_with_fake_engine()
    with pytest.raises(KeyError, match="unknown scenario Actor"):
        manager.command_actor("missing", ActorCommand.longitudinal(-4.0))
```

- [ ] **Step 10: Run Actor manager tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_actor_manager.py -q`

Expected: FAIL because adapter dataclasses and `ScenarioActorManager` do not exist.

- [ ] **Step 11: Implement the adapter contracts and thin MetaDrive manager**

Use frozen dataclasses with finite-value validation. `ScenarioActorManager` subclasses MetaDrive `BaseManager`, records all spawned IDs in the inherited ownership map, applies pending commands in `before_step()`, calls Actor state refresh in `after_step()`, and relies on `BaseManager.before_reset()` for cleanup.

Register it after `super().setup_engine()`:

```python
def setup_engine(self) -> None:
    super().setup_engine()
    self.engine.register_manager("scenario_actor_manager", ScenarioActorManager())
```

Expose typed forwarding methods on `ControlMetaDriveEnv`; runtimes must never access `engine` directly.

- [ ] **Step 12: Run Actor manager tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_actor_manager.py -q`

Expected: PASS.

- [ ] **Step 13: Write failing Nominal and Lead Brake runtime tests**

Use `tests/unit/scenarios/fakes.py` to verify spawn geometry, trigger timing, deceleration choice, success window, collision failure, and internal Actor disappearance:

```python
def test_lead_brake_triggers_sampled_deceleration() -> None:
    runtime, environment, state = reset_lead_brake(trigger_s=1.0, deceleration_mps2=4.0)
    runtime.before_step(environment, state, step_index=10)
    assert environment.commands[-1] == ("lead-brake", ActorCommand.longitudinal(-4.0))


def test_lead_brake_succeeds_after_survival_window() -> None:
    runtime, environment, state = reset_lead_brake(trigger_s=1.0, survival_s=4.0)
    transition = runtime.after_step(
        environment, state, step_index=50, raw_info=no_collision_info()
    )
    assert transition.outcome == ScenarioStepResult(success=True, failure=False)


def test_missing_spawned_actor_is_internal_error() -> None:
    runtime, environment, state = reset_lead_brake()
    environment.remove("lead-brake")
    with pytest.raises(RuntimeError, match="missing scenario Actor"):
        runtime.before_step(environment, state, step_index=1)
```

- [ ] **Step 14: Run runtime tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_lead_brake.py tests/unit/scenarios/test_manager.py -q`

Expected: FAIL because concrete runtimes are absent.

- [ ] **Step 15: Implement Nominal, Lead Brake, factory wiring, and scenario info**

The Lead Brake state contains sampled `initial_gap_m`, `lead_speed_mps`, `trigger_step`, `deceleration_mps2`, `success_step`, and `actor_id`. Convert seconds to steps using the validated decision interval. Level 1 samples mild deceleration; Levels 2-3 sample severe deceleration. `after_step()` treats `crash_vehicle=True` as scenario failure only when the spawned lead Actor is still present and the ego collision flag is set.

Construct the default Phase 5 factory from `AppConfig`, while `scenario_id != "phase5"` continues to use `NoOpScenarioRuntime`. Add reset and step info fields `scenario_id`, `difficulty_level`, `scenario_parameters`, `scenario_success`, and `scenario_failure`.

- [ ] **Step 16: Run Task 1 unit and real integration tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/config tests/unit/scenarios tests/unit/envs/test_multi_agent_speed_env.py -q`

Expected: PASS.

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration`

Expected: PASS with a real Lead Brake Actor present, finite observations/rewards, deterministic identical-seed prefix, and no Actor leakage after reset.

- [ ] **Step 17: Run static checks and commit Task 1**

Run: `.venv\Scripts\python.exe -m ruff check src tests`

Expected: PASS.

Run: `.venv\Scripts\python.exe -m mypy src`

Expected: PASS.

Commit:

```bash
git add configs src/mad_driving/config src/mad_driving/scenarios src/mad_driving/envs tests/unit/config tests/unit/scenarios tests/unit/envs/test_multi_agent_speed_env.py tests/integration/test_phase5_metadrive_headless.py
git commit -m "feat: add seeded lead-brake scenario"
```

### Task 2: Cut-in

**Files:**
- Create `configs/scenarios/cut_in.yaml`.
- Create `src/mad_driving/scenarios/cut_in.py`.
- Modify `src/mad_driving/config/models.py`, `src/mad_driving/scenarios/manager.py`, and `src/mad_driving/scenarios/__init__.py`.
- Create `tests/unit/scenarios/test_cut_in.py`.
- Extend `tests/integration/test_phase5_metadrive_headless.py`.

**Interfaces:**
- Consumes the Task 1 Actor adapter and `ScenarioParameterSampler`.
- Produces concrete scenario ID `cut_in` and cubic trajectory function `smoothstep(progress: float) -> float`.

- [ ] **Step 1: Write failing Cut-in parameter and trajectory tests**

```python
@pytest.mark.parametrize(("progress", "expected"), [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
def test_smoothstep_endpoints_and_midpoint(progress: float, expected: float) -> None:
    assert smoothstep(progress) == pytest.approx(expected)


def test_cut_in_moves_from_adjacent_lane_to_ego_lane() -> None:
    runtime, environment, state = reset_cut_in(trigger_s=1.0, merge_s=2.0)
    runtime.before_step(environment, state, step_index=10)
    start = environment.commands[-1][1]
    runtime.before_step(environment, state, step_index=20)
    middle = environment.commands[-1][1]
    runtime.before_step(environment, state, step_index=30)
    end = environment.commands[-1][1]
    assert abs(start.lateral_m) > abs(middle.lateral_m) > abs(end.lateral_m)
    assert end.lateral_m == pytest.approx(0.0)
```

- [ ] **Step 2: Run Cut-in tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_cut_in.py -q`

Expected: FAIL because `cut_in.py` does not exist.

- [ ] **Step 3: Implement sampled Cut-in and cubic motion**

Implement the bounded curve:

```python
def smoothstep(progress: float) -> float:
    bounded = min(max(progress, 0.0), 1.0)
    return bounded * bounded * (3.0 - 2.0 * bounded)
```

Sample initial lead 20-40 m, trigger 1-3 s, merge duration 1.5-3 s, and speed fraction 0.75-1.05. Choose available adjacent lane indices in sorted canonical order. Keep longitudinal velocity continuous and command lateral position as `source_lateral_m * (1.0 - smoothstep(progress))`. Mark success after merge plus 3 s survival; mark ego collision with the Actor as failure; raise if no adjacent lane exists.

- [ ] **Step 4: Run Cut-in unit tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_cut_in.py tests/unit/scenarios/test_manager.py -q`

Expected: PASS.

- [ ] **Step 5: Add and run real deterministic Cut-in smoke**

The integration test must load `configs/base.yaml` plus `configs/scenarios/cut_in.yaml`, assert the Actor begins in an adjacent lane, reaches the ego lane, preserves finite state, repeats the same first 20 positions for the same seed, and is absent after the next reset.

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration -k cut_in`

Expected: PASS.

- [ ] **Step 6: Run checks and commit Task 2**

Run: `.venv\Scripts\python.exe -m ruff check src tests`

Expected: PASS.

Run: `.venv\Scripts\python.exe -m mypy src`

Expected: PASS.

Commit:

```bash
git add configs/scenarios/cut_in.yaml src/mad_driving/scenarios tests/unit/scenarios/test_cut_in.py tests/integration/test_phase5_metadrive_headless.py
git commit -m "feat: add seeded cut-in scenario"
```

### Task 3: Occluded Crossing Actor

**Files:**
- Create `configs/scenarios/occluded_crossing.yaml`.
- Create `src/mad_driving/scenarios/occluded_crossing.py`.
- Modify `src/mad_driving/config/models.py`, `src/mad_driving/scenarios/manager.py`, and `src/mad_driving/scenarios/__init__.py`.
- Create `tests/unit/scenarios/test_occluded_crossing.py`.
- Extend `tests/integration/test_phase5_metadrive_headless.py`.
- Modify `src/mad_driving/world_model/snapshot_builder.py` only if required by the stable Cyclist ID integration test.

**Interfaces:**
- Consumes Task 1 crossing-Actor and occluder spawn methods.
- Produces concrete scenario ID `occluded_crossing` and `ScenarioObservationContext` with one `OcclusionRegion`, current conflict distance, and a complete visible-ID allowlist.

- [ ] **Step 1: Write failing occlusion-boundary and outcome tests**

```python
def test_crossing_actor_is_hidden_but_privileged_before_reveal() -> None:
    runtime, environment, state = reset_crossing(actor_lateral_m=8.0, reveal_lateral_m=3.0)
    context = runtime.observation_context(state)
    assert "crossing-cyclist" not in context.visible_actor_ids
    assert "static-occluder" in context.visible_actor_ids


def test_crossing_actor_becomes_visible_at_boundary() -> None:
    runtime, environment, state = reset_crossing(actor_lateral_m=2.9, reveal_lateral_m=3.0)
    transition = runtime.after_step(environment, state, step_index=20, raw_info=no_collision_info())
    context = runtime.observation_context(transition.state)
    assert "crossing-cyclist" in context.visible_actor_ids


def test_crossing_collision_is_failure() -> None:
    runtime, environment, state = reset_crossing()
    transition = runtime.after_step(
        environment, state, step_index=20, raw_info={"crash_human": True}
    )
    assert transition.outcome == ScenarioStepResult(success=False, failure=True)
```

- [ ] **Step 2: Run Crossing tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_occluded_crossing.py -q`

Expected: FAIL because the runtime does not exist.

- [ ] **Step 3: Implement cyclist, occluder, geometry, and logical visibility**

Sample conflict distance 20-40 m, crossing start offset 6-12 m, speed 2-6 m/s, trigger 1-3 s, and survival 2 s. Spawn a MetaDrive Cyclist with a stable name and a static occluder at least 0.5 m beyond the lane edge. Also spawn a visible same-lane lead vehicle 35-55 m ahead at 0.80-1.00 of ego speed; it maintains speed and provides the required second hazard candidate. Keep the cyclist in privileged simulator truth from reset. Build the visible allowlist from all current scenario and traffic Actor IDs, excluding only `crossing-cyclist` until its absolute lateral distance is at or below the reveal boundary.

Compute current conflict distance from ego lane coordinates on every context request; do not expose the hidden Actor's kinematics through `ScenarioObservationContext`. Mark success when the cyclist clears the ego collision corridor and survives 2 s. Mark `crash_human=True` as failure.

- [ ] **Step 4: Run Crossing unit and world-model tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_occluded_crossing.py tests/unit/world_model/test_snapshot_builder.py tests/unit/agents/test_hazard.py -q`

Expected: PASS, including hidden Actor absent from `visible_actors` and present in privileged `all_actors` as `visible=False, occluded=True`.

- [ ] **Step 5: Add and run real occluded-crossing smoke**

The real test must verify `crash_human` classification support, stable Actor ID, hidden-before-reveal behavior, visible-after-reveal behavior, finite oracle TTC, deterministic identical-seed trajectory, and cleanup across reset.

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py -q -m integration -k occluded`

Expected: PASS.

- [ ] **Step 6: Run checks and commit Task 3**

Run: `.venv\Scripts\python.exe -m ruff check src tests`

Expected: PASS.

Run: `.venv\Scripts\python.exe -m mypy src`

Expected: PASS.

Commit:

```bash
git add configs/scenarios/occluded_crossing.yaml src/mad_driving/scenarios src/mad_driving/world_model/snapshot_builder.py tests/unit/scenarios/test_occluded_crossing.py tests/unit/world_model/test_snapshot_builder.py tests/integration/test_phase5_metadrive_headless.py
git commit -m "feat: add seeded occluded-crossing scenario"
```

### Task 4: Curriculum, provenance, resume, and final verification

**Files:**
- Create/modify every file listed under “Curriculum, provenance, and completion”.

**Interfaces:**
- Produces immutable `CurriculumState(level: int, consecutive_passes: int, evaluations: int)`.
- Produces `CurriculumController.observe(role: EnvironmentRole, successes: int, collisions: int, episodes: int) -> CurriculumState`.
- Produces `CurriculumEvalCallback`, extending fixed-seed validation and broadcasting levels through `VecEnv.env_method("set_difficulty_level", level)`.
- Expands episode seed artifact records to exact fields `role`, `worker_index`, three seed identities, `scenario_id`, `difficulty_level`, and `scenario_parameters`.
- Adds `DecisionTrace.scenario_id: str` and `DecisionTrace.difficulty_level: int` whenever episode metadata is present.

- [ ] **Step 1: Write failing curriculum state-machine tests**

```python
def test_automatic_curriculum_advances_after_two_passing_validations() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(0, 0, 0))
    first = controller.observe("validation", successes=4, collisions=0, episodes=5)
    assert first == CurriculumState(level=0, consecutive_passes=1, evaluations=1)
    second = controller.observe("validation", successes=5, collisions=0, episodes=5)
    assert second == CurriculumState(level=1, consecutive_passes=0, evaluations=2)


def test_test_role_cannot_drive_curriculum() -> None:
    controller = CurriculumController(automatic_config(), CurriculumState(0, 0, 0))
    with pytest.raises(ValueError, match="test role"):
        controller.observe("test", successes=5, collisions=0, episodes=5)


def test_fixed_curriculum_never_advances() -> None:
    controller = CurriculumController(fixed_config(level=2), CurriculumState(2, 0, 0))
    assert controller.observe("validation", 5, 0, 5).level == 2
```

- [ ] **Step 2: Run curriculum tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/training/test_curriculum.py -q`

Expected: FAIL because `training/curriculum.py` does not exist.

- [ ] **Step 3: Implement the immutable curriculum controller**

Validate integer counts, `0 <= successes <= episodes`, `0 <= collisions <= episodes`, and `episodes > 0`. A passing evaluation requires both configured thresholds. Advance by one level only, cap at 3, reset consecutive passes after advancement, and never regress.

```python
passed = successes / episodes >= config.success_rate_threshold and (
    collisions / episodes <= config.collision_rate_threshold
)
next_passes = state.consecutive_passes + 1 if passed else 0
advance = next_passes >= config.consecutive_evaluations and state.level < 3
```

- [ ] **Step 4: Run curriculum tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/training/test_curriculum.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing callback metric and reset-boundary broadcast tests**

```python
def test_callback_broadcasts_new_level_after_passing_eval() -> None:
    callback, training_env, validation_env = callback_with_metrics(
        terminal_infos=[
            {"scenario_success": True, "collision_occurred": False},
            {"scenario_success": True, "collision_occurred": False},
        ],
        consecutive_evaluations=1,
    )
    callback.run_scheduled_evaluation()
    assert training_env.calls[-1] == ("set_difficulty_level", 1)
    assert validation_env.calls[-1] == ("set_difficulty_level", 1)


def test_level_change_does_not_mutate_active_episode() -> None:
    env = configured_fake_speed_env(level=0)
    env.reset(seed=42)
    env.set_difficulty_level(2)
    assert env.active_difficulty_level == 0
    env.reset(seed=43)
    assert env.active_difficulty_level == 2
```

- [ ] **Step 6: Run callback tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/training/test_callbacks.py tests/unit/envs/test_multi_agent_speed_env.py -q`

Expected: FAIL because validation metrics and level broadcast are not implemented.

- [ ] **Step 7: Implement callback capture and level broadcast**

Extend `SeededEvalCallback._log_success_callback()` to collect one terminal record per completed evaluation episode from `locals_["info"]` and `locals_["done"]`. After `super()._on_step()` completes a scheduled evaluation, pass exact counts to `CurriculumController`, atomically persist the returned state, and call `env_method` on both training and validation VecEnvs only when the level changes.

Reject missing or non-boolean scenario outcome fields. Keep the existing validation reseed immediately before each evaluation.

- [ ] **Step 8: Run callback tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/training/test_callbacks.py tests/unit/envs/test_multi_agent_speed_env.py -q`

Expected: PASS.

- [ ] **Step 9: Write failing provenance, DecisionTrace, and resume tests**

```python
def test_seed_artifact_records_scenario_generation_conditions(tmp_path: Path) -> None:
    record = read_single_episode_record(run_wrapped_reset(tmp_path))
    assert record["scenario_id"] == "lead_brake"
    assert record["difficulty_level"] == 1
    assert record["scenario_parameters"]["initial_gap_m"] >= 35.0


def test_decision_trace_requires_complete_scenario_metadata() -> None:
    with pytest.raises(ValueError, match="scenario trace metadata must be complete"):
        replace(make_trace(), scenario_id="lead_brake", difficulty_level=None)


def test_resume_restores_curriculum_state(tmp_path: Path) -> None:
    parent = make_parent_run(tmp_path, CurriculumState(2, 1, 7))
    result = run_training(resume_from=parent.checkpoint, total_timesteps=8)
    assert read_curriculum_state(result.run_dir).level == 2
    assert read_curriculum_state(result.run_dir).evaluations >= 7
```

- [ ] **Step 10: Run provenance tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/training/test_episode_seeds.py tests/unit/training/test_metadata.py tests/unit/training/test_train.py tests/unit/interfaces/test_models.py -q`

Expected: FAIL on old artifact schema, missing curriculum metadata, and missing trace fields.

- [ ] **Step 11: Implement schema v3, research contract v5, and atomic resume state**

Set `EPISODE_SEED_ARTIFACT_SCHEMA_VERSION = 3` in both artifact and metadata modules and `RESEARCH_CONTRACT_VERSION = 5`. Validate exact record fields and recursively JSON-safe finite parameters. Preserve existing descriptor identity, append/fsync, strict parser, count, and SHA-256 behavior.

Write `curriculum_state.yaml` through a same-directory temporary file, flush and fsync it, then `os.replace()`. Include its SHA-256 and immutable values in `run_metadata.json`. Resume requires the parent curriculum state and rejects a missing file, hash mismatch, malformed state, level outside 0-3, or configuration incompatibility.

Extend `DecisionTrace.__post_init__()` so scenario ID and level are both absent or both valid; when seed metadata is present in Phase 5 they are required.

- [ ] **Step 12: Run provenance tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/training/test_episode_seeds.py tests/unit/training/test_metadata.py tests/unit/training/test_train.py tests/unit/interfaces/test_models.py -q`

Expected: PASS.

- [ ] **Step 13: Add Level 0-3 and PPO integration smokes**

Add real tests that run one short fixed-level episode for every level, verify every selected scenario is allowed at that level, and verify no test seed appears in train/validation artifacts. Add an 8-timestep PPO smoke in fixed Level 1 mode and an automatic-curriculum smoke with one scheduled validation; assert finite model artifacts, persisted curriculum state, scenario fields in JSONL, and contract version 5.

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_phase5_metadrive_headless.py tests/integration/test_rl_metadrive_headless.py tests/integration/test_ppo_checkpoint.py -q -m integration`

Expected: PASS.

- [ ] **Step 14: Update README and verify every CLI help path**

Document base-plus-overlay loading, the three scenario overlay paths, fixed versus automatic curriculum, seed identities, and the fact that the 24D observation is unchanged. Include exact commands:

```powershell
python -m mad_driving.cli.train --config configs/base.yaml --smoke
python -m pytest tests/integration/test_phase5_metadrive_headless.py -m integration -q
```

Run: `.venv\Scripts\python.exe -m mad_driving.cli.train --help`

Expected: exit code 0 and documented config/smoke options.

- [ ] **Step 15: Run final complete verification**

Run: `.venv\Scripts\python.exe -m ruff check src tests`

Expected: PASS.

Run: `.venv\Scripts\python.exe -m mypy src`

Expected: PASS.

Run: `.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q`

Expected: complete suite PASS and total coverage at least 90%.

Run each Phase 5 real smoke twice with the same seed and compare selected scenario, sampled parameters, Actor trajectory prefix, outcomes, and DecisionTrace metadata byte-for-byte.

Expected: every repeated pair is identical.

- [ ] **Step 16: Commit Task 4**

```bash
git add src/mad_driving/training src/mad_driving/interfaces/decision_trace.py src/mad_driving/envs/multi_agent_speed_env.py tests/unit/training tests/unit/interfaces/test_models.py tests/unit/envs/test_multi_agent_speed_env.py tests/integration README.md
git commit -m "feat: add reproducible phase 5 curriculum"
```

After this commit, inspect `git status --short`, `git diff feat/phase4-rl-environment...HEAD --check`, and the four Phase 5 commits before pushing the branch and opening stacked PR #5 with base `feat/phase4-rl-environment`.
