# Phase 4 RL Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the fixed Phase 3 decision pipeline into a reproducible 24-dimensional Gymnasium environment and train only its Coordinator with PPO, producing loadable checkpoints and TensorBoard events.

**Architecture:** Keep Observation and Reward as simulator-independent components. `MultiAgentSpeedEnv` alone owns MetaDrive, Agent, Shield, and episode lifecycle; PPO sees only `Box(24)` observations and `Discrete(4)` requested actions. Training code constructs vector environments and callbacks but never reaches into simulator objects or learns Agent, Shield, PID, or steering parameters.

**Tech Stack:** Python 3.11.9, MetaDrive 0.4.3, Gymnasium 1.3.0, NumPy 1.26.4, Pydantic 2.11.7, Stable-Baselines3 2.9.0, PyTorch 2.8.0 CPU, TensorBoard 2.19.0, pytest 8.4.1, Ruff 0.12.4, mypy 1.16.1.

## Global Constraints

- `docs/multi_agent_driving_mvp_spec.md` is the highest-level requirement; `docs/superpowers/specs/2026-07-20-phase4-rl-environment-design.md` fixes only values the MVP leaves unspecified.
- Implement only Phase 4: fixed 24-dimensional Observation, Reward, Gymnasium wrapper/checker, PPO, checkpoint/resume, and TensorBoard.
- Preserve every Phase 1-3 CLI and the 267-test clean baseline.
- Keep the three specialized scenarios, ScenarioManager, curriculum, train/eval seed sets, baselines, ablations, JSONL, CSV, plots, GIF, evaluation CLI, and comparison CLI out of Phase 4.
- PPO may optimize only Coordinator policy parameters. Agents, Critic, Safety Shield, lane steering, and speed PID remain fixed deterministic modules.
- Observation is always shape `(24,)`, dtype `numpy.float32`, finite, and within `[-1, 1]`; action space is `gymnasium.spaces.Discrete(4)`.
- `info["reward_components"]` and `DecisionTrace.reward_components` contain the same ten signed finite values and sum exactly to the scalar reward.
- Unknown config keys, scalar coercion in Phase 4 models, invalid ranges, and non-finite settings fail at startup.
- Use TDD for every production behavior: add a test, observe the expected RED failure, add minimal code, observe GREEN, then refactor.
- All simulator and vector environments close on success and failure. Simulator exceptions become truncated episodes instead of killing a PPO rollout.
- Same reset seed and same action sequence produce the same initial Observation and deterministic Agent outputs.
- Keep Python `>=3.11,<3.12`, direct dependencies pinned, branch coverage at least 80%, Ruff, Ruff format, and mypy strict.
- Record MetaDrive/SB3 API observations and every compatibility adjustment before changing an assumed binding.

---

## File map

```text
configs/
├─ base.yaml                                # explicit Phase 4 defaults
└─ train.yaml                               # complete training config

src/mad_driving/
├─ config/models.py                         # strict Observation/Reward/PPO config
├─ coordinator/
│  ├─ __init__.py
│  └─ observation.py                        # fixed 24D builder
├─ envs/
│  ├─ __init__.py
│  ├─ multi_agent_speed_env.py              # Gymnasium wrapper + existing boundaries
│  └─ reward.py                             # ten-component transition reward
├─ training/
│  ├─ __init__.py
│  ├─ callbacks.py                          # reward-component TensorBoard logging
│  └─ train.py                              # PPO/vector env/checkpoint lifecycle
└─ cli/
   └─ train.py                              # command-line boundary

tests/
├─ unit/config/test_rl_config.py
├─ unit/coordinator/test_observation.py
├─ unit/envs/test_reward.py
├─ unit/envs/test_multi_agent_speed_env.py
├─ unit/training/test_callbacks.py
├─ unit/training/test_train.py
├─ unit/cli/test_train.py
├─ integration/test_rl_metadrive_headless.py
└─ integration/test_ppo_checkpoint.py
```

---

### Task 1: Strict Phase 4 configuration

**Files:**
- Modify: `src/mad_driving/config/models.py`
- Modify: `configs/base.yaml`
- Create: `configs/train.yaml`
- Create: `tests/unit/config/test_rl_config.py`

**Interfaces:**
- Produces `ObservationConfig`, `RewardConfig`, and `PPOConfig`.
- Extends `AppConfig.observation`, `AppConfig.reward`, and `AppConfig.training` with default factories.
- All later tasks consume these frozen models; no later component reads YAML directly.

- [x] **Step 1: Write failing default, explicit-value, and strictness tests**

```python
def test_phase4_defaults_match_specification() -> None:
    config = AppConfig.model_validate(minimum_app_config())
    assert config.observation.max_ttc_s == 10.0
    assert config.reward.progress_per_meter == 0.10
    assert config.reward.collision_crossing_actor == 500.0
    assert config.training.algorithm == "PPO"
    assert config.training.n_steps == 2048
    assert config.training.smoke_timesteps == 5_000
    assert config.training.total_timesteps == 500_000


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (ObservationConfig, {"max_ttc_s": "10.0"}),
        (RewardConfig, {"near_miss_max": float("nan")}),
        (PPOConfig, {"batch_size": 0}),
        (PPOConfig, {"algorithm": "DQN"}),
    ],
)
def test_phase4_models_reject_invalid_or_coerced_values(model, payload) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_ppo_rollout_size_must_be_divisible_by_batch() -> None:
    with pytest.raises(ValidationError, match="batch_size"):
        PPOConfig(n_steps=10, num_envs=1, batch_size=6)
```

- [x] **Step 2: Run the config test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/config/test_rl_config.py -v`

Expected: collection fails because the three Phase 4 config classes do not exist.

- [x] **Step 3: Implement strict frozen models and validators**

```python
class ObservationConfig(StrictTypedFrozenModel):
    max_speed_mps: FiniteFloat = Field(default=40.0, gt=0.0)
    max_abs_acceleration_mps2: FiniteFloat = Field(default=10.0, gt=0.0)
    max_abs_lane_offset_m: FiniteFloat = Field(default=3.5, gt=0.0)
    max_ttc_s: FiniteFloat = Field(default=10.0, gt=0.0)
    max_abs_stopping_margin_m: FiniteFloat = Field(default=50.0, gt=0.0)


class RewardConfig(StrictTypedFrozenModel):
    progress_per_meter: FiniteFloat = Field(default=0.10, ge=0.0)
    arrival: FiniteFloat = Field(default=100.0, ge=0.0)
    collision_vehicle: FiniteFloat = Field(default=200.0, ge=0.0)
    collision_crossing_actor: FiniteFloat = Field(default=500.0, ge=0.0)
    near_miss_max: FiniteFloat = Field(default=50.0, ge=0.0)
    near_miss_ttc_s: FiniteFloat = Field(default=3.0, gt=0.0)
    offroad: FiniteFloat = Field(default=100.0, ge=0.0)
    hard_rule_violation: FiniteFloat = Field(default=100.0, ge=0.0)
    jerk_scale: FiniteFloat = Field(default=0.05, ge=0.0)
    unnecessary_brake_scale: FiniteFloat = Field(default=0.20, ge=0.0)
    unnecessary_brake_severity_threshold: FiniteFloat = Field(default=0.25, ge=0.0, le=1.0)
    unnecessary_brake_safe_ttc_s: FiniteFloat = Field(default=5.0, gt=0.0)
    unnecessary_brake_lookahead_steps: PositiveInt = 3
    standstill_per_second: FiniteFloat = Field(default=0.50, ge=0.0)
    standstill_speed_mps: FiniteFloat = Field(default=0.10, ge=0.0)
    shield_intervention: FiniteFloat = Field(default=2.0, ge=0.0)


class PPOConfig(StrictTypedFrozenModel):
    algorithm: Literal["PPO"] = "PPO"
    policy: Literal["MlpPolicy"] = "MlpPolicy"
    learning_rate: FiniteFloat = Field(default=0.0003, gt=0.0)
    n_steps: PositiveInt = 2048
    batch_size: PositiveInt = 64
    n_epochs: PositiveInt = 10
    gamma: FiniteFloat = Field(default=0.99, gt=0.0, le=1.0)
    gae_lambda: FiniteFloat = Field(default=0.95, ge=0.0, le=1.0)
    clip_range: FiniteFloat = Field(default=0.2, gt=0.0)
    ent_coef: FiniteFloat = Field(default=0.01, ge=0.0)
    vf_coef: FiniteFloat = Field(default=0.5, ge=0.0)
    max_grad_norm: FiniteFloat = Field(default=0.5, gt=0.0)
    seed: int = Field(default=42, ge=0)
    smoke_timesteps: PositiveInt = 5_000
    total_timesteps: PositiveInt = 500_000
    num_envs: PositiveInt = 1
    checkpoint_interval_steps: PositiveInt = 10_000
    eval_interval_steps: PositiveInt = 10_000
    eval_episodes: PositiveInt = 5
    run_root: str = Field(default="runs", min_length=1)
```

Add an `after` validator for rollout divisibility and add the three defaulted fields to `AppConfig`.
Write every selected value explicitly in `base.yaml` and make `train.yaml` a complete standalone config.

- [x] **Step 4: Run config and loader suites and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/config -v`

Expected: all config tests pass, including legacy Phase 1-3 payloads.

- [x] **Step 5: Commit Task 1**

```powershell
git add src/mad_driving/config/models.py configs/base.yaml configs/train.yaml tests/unit/config/test_rl_config.py
git commit -m "feat: add phase 4 rl configuration"
```

---

### Task 2: Fixed 24-dimensional Observation

**Files:**
- Create: `src/mad_driving/coordinator/observation.py`
- Modify: `src/mad_driving/coordinator/__init__.py`
- Create: `tests/unit/coordinator/test_observation.py`

**Interfaces:**
- Produces `ObservationBuilder(config: ObservationConfig)`.
- Produces `ObservationBuilder.build(snapshot, claims, review) -> NDArray[np.float32]`.
- Uses `target_speed_mps` for index 1 and exact index meanings from the design.

- [x] **Step 1: Write the failing exact-layout test**

```python
def test_observation_has_exact_layout_dtype_and_bounds() -> None:
    obs = ObservationBuilder(ObservationConfig()).build(
        make_snapshot(),
        (make_nominal_claim(), make_hazard_claim(), make_rule_claim()),
        make_review(),
    )
    assert obs.shape == (24,)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs, EXPECTED_24_VALUES)
    assert np.isfinite(obs).all()
    assert (obs >= -1.0).all() and (obs <= 1.0).all()
```

Add focused tests for each normalization boundary, `None` TTC=`1.0`, clipped huge values,
duplicate agent IDs, invalid defensive input, and deterministic repeated output.

- [x] **Step 2: Write the failing conservative-missing-claim test**

```python
def test_missing_claims_become_finite_safe_side_features() -> None:
    obs = ObservationBuilder(ObservationConfig()).build(make_snapshot(), (), make_review())
    assert obs[6:10].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert obs[10:15].tolist() == [0.0, -1.0, 1.0, 0.0, 0.0]
    assert obs[15:18].tolist() == [0.0, 1.0, 1.0]
    assert np.isfinite(obs).all()
```

- [x] **Step 3: Run the Observation tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/coordinator/test_observation.py -v`

Expected: import failure because `ObservationBuilder` does not exist.

- [x] **Step 4: Implement minimal normalization and exact index assembly**

```python
class ObservationBuilder:
    def __init__(self, config: ObservationConfig) -> None:
        self._config = config

    def build(
        self,
        snapshot: SceneSnapshot,
        claims: Sequence[RiskClaim],
        review: CriticReview,
    ) -> NDArray[np.float32]:
        # validate defensive boundaries, index claims once, assemble exactly 24 values
        values = np.asarray([...], dtype=np.float32)
        if values.shape != (24,) or not np.isfinite(values).all():
            raise ValueError("observation must contain 24 finite values")
        return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)
```

Use small `_unit`, `_signed`, `_ttc`, `_speed`, and `_claim_index` helpers; do not expose a
generic normalization framework.

- [x] **Step 5: Run Observation and property-loop tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/coordinator/test_observation.py -v`

Expected: all tests pass for exact values, 1,000 finite randomized boundary inputs, and determinism.

- [x] **Step 6: Commit Task 2**

```powershell
git add src/mad_driving/coordinator tests/unit/coordinator/test_observation.py
git commit -m "feat: build fixed coordinator observations"
```

---

### Task 3: Ten-component Reward

**Files:**
- Create: `src/mad_driving/envs/reward.py`
- Create: `tests/unit/envs/test_reward.py`

**Interfaces:**
- Produces immutable `RewardContext` with previous/next Snapshot, post-step claims, executed action,
  shield intervention, arrival, collision kind, and fallback decision interval.
- Produces immutable `RewardResult(total, components)`.
- Produces stateful `RewardCalculator(config).reset()` and `.calculate(context)`.

- [x] **Step 1: Write one failing test per Reward component**

```python
def test_reward_components_are_signed_and_sum_to_total() -> None:
    result = RewardCalculator(RewardConfig()).calculate(make_context())
    assert tuple(result.components) == EXPECTED_COMPONENT_KEYS
    assert result.total == pytest.approx(sum(result.components.values()))
    assert all(math.isfinite(value) for value in result.components.values())


@pytest.mark.parametrize(
    ("collision_kind", "expected"),
    [("vehicle", -200.0), ("crossing_actor", -500.0), (None, 0.0)],
)
def test_collision_penalty_uses_collision_kind(collision_kind, expected) -> None:
    result = calculator().calculate(make_context(collision_kind=collision_kind))
    assert result.components["collision_penalty"] == expected
```

Separate tests cover forward-only progress, one-shot arrival, continuous near-miss at TTC
`0`, midpoint, threshold and above, off-road, rule violation, jerk with fallback `dt`, standstill,
and Shield intervention.

- [x] **Step 2: Write the failing unnecessary-brake lookahead/reset tests**

```python
def test_unnecessary_brake_penalty_starts_after_safe_lookahead() -> None:
    calc = RewardCalculator(RewardConfig(unnecessary_brake_lookahead_steps=3))
    penalties = [
        calc.calculate(make_safe_braking_context()).components["unnecessary_brake_penalty"]
        for _ in range(4)
    ]
    assert penalties == [0.0, 0.0, -0.2, -0.2]
    calc.reset()
    assert calc.calculate(make_safe_braking_context()).components[
        "unnecessary_brake_penalty"
    ] == 0.0
```

Also prove any dangerous post-step event clears the streak.

- [x] **Step 3: Run Reward tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/envs/test_reward.py -v`

Expected: import failure because the Reward module does not exist.

- [x] **Step 4: Implement the minimal Reward state machine**

```python
@dataclass(frozen=True)
class RewardResult:
    total: float
    components: dict[str, float]


class RewardCalculator:
    def reset(self) -> None:
        self._safe_brake_streak = 0
        self._arrival_rewarded = False

    def calculate(self, context: RewardContext) -> RewardResult:
        components = {
            "progress_reward": ...,
            "arrival_reward": ...,
            "collision_penalty": ...,
            "near_miss_penalty": ...,
            "offroad_penalty": ...,
            "rule_violation_penalty": ...,
            "jerk_penalty": ...,
            "unnecessary_brake_penalty": ...,
            "standstill_penalty": ...,
            "shield_intervention_penalty": ...,
        }
        return RewardResult(total=sum(components.values()), components=components)
```

Validate every computed value before returning and copy the components mapping in
`RewardResult.__post_init__`.

- [x] **Step 5: Run Reward tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/envs/test_reward.py -v`

Expected: all ten components, reset behavior, continuity, and finite-value tests pass.

- [x] **Step 6: Commit Task 3**

```powershell
git add src/mad_driving/envs/reward.py tests/unit/envs/test_reward.py
git commit -m "feat: calculate phase 4 driving rewards"
```

---

### Task 4: Gymnasium `MultiAgentSpeedEnv`

**Files:**
- Modify: `src/mad_driving/envs/multi_agent_speed_env.py`
- Modify: `src/mad_driving/envs/__init__.py`
- Create: `tests/unit/envs/test_multi_agent_speed_env.py`

**Interfaces:**
- Adds `MultiAgentSpeedEnv(gym.Env[NDArray[np.float32], int])` without removing existing
  `DrivingEnvironment`, `SmokeResult`, or `ControlSmokeResult`.
- Constructor accepts `AppConfig` and injectable environment/suite/shield/builder/reward/observation
  factories for real and fake tests.
- Public Gym API is exactly `reset(*, seed=None, options=None)` and `step(action)`.

- [x] **Step 1: Write failing space/reset/seed tests with a complete fake simulator**

```python
def test_env_exposes_fixed_spaces_and_seeded_reset() -> None:
    env = make_env()
    try:
        first, info = env.reset(seed=123)
        second, _ = env.reset(seed=123)
        assert env.action_space == gym.spaces.Discrete(4)
        assert env.observation_space.shape == (24,)
        assert env.observation_space.dtype == np.float32
        assert env.observation_space.contains(first)
        np.testing.assert_array_equal(first, second)
        assert info["seed"] == 123
    finally:
        env.close()
```

- [x] **Step 2: Write failing decision-pipeline and trace tests**

Prove that requested action reaches Shield, only executed action reaches simulator, next claims build
next Observation, scalar reward equals component sum, and `DecisionTrace` uses pre-step claims/review
plus post-step Reward components.

```python
obs, reward, terminated, truncated, info = env.step(DrivingAction.KEEP)
assert fake_sim.actions == [int(DrivingAction.STOP)]
assert info["requested_action"] == int(DrivingAction.KEEP)
assert info["executed_action"] == int(DrivingAction.STOP)
assert info["decision_trace"].reward_components == info["reward_components"]
assert reward == pytest.approx(sum(info["reward_components"].values()))
```

- [x] **Step 3: Write failing lifecycle/error tests**

Cover step-before-reset, invalid action, idempotent close, reset after close, raw terminated/truncated
mapping, arrival/crash info mapping, Agent exception fallback, Observation failure causing STOP/truncation,
simulator step exception returning `truncated=True`, and simulator recreation on next reset.

- [x] **Step 4: Run the wrapper tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/envs/test_multi_agent_speed_env.py -v`

Expected: import failure because concrete `MultiAgentSpeedEnv` does not exist.

- [x] **Step 5: Implement the minimal Gymnasium wrapper**

```python
class MultiAgentSpeedEnv(gym.Env[NDArray[np.float32], int]):
    metadata = {"render_modes": []}

    def __init__(self, config: AppConfig, ...) -> None:
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32
        )
        # store factories; construct one simulator; initialize episode state to None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # recreate failed/closed simulator, reset deterministic components, build initial state

    def step(self, action):
        # validate state/action, Shield requested action, advance simulator, reward, trace, next obs

    def close(self) -> None:
        # close at most once per simulator instance and clear episode state
```

Move Phase 3 `_fallback_analysis` and `_analyze_safely` to a shared simulator-independent helper only
if needed by both the old smoke and wrapper; preserve the old CLI behavior and tests exactly.

- [x] **Step 6: Run wrapper plus Phase 3 smoke suites and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/envs/test_multi_agent_speed_env.py tests/unit/cli/test_control_smoke.py -v`

Expected: all new lifecycle tests and all existing control smoke tests pass.

- [x] **Step 7: Commit Task 4**

```powershell
git add src/mad_driving/envs src/mad_driving/agents src/mad_driving/cli/control_smoke.py tests/unit/envs/test_multi_agent_speed_env.py
git commit -m "feat: add gymnasium multi-agent speed environment"
```

---

### Task 5: Real MetaDrive Gymnasium compliance

**Files:**
- Create: `tests/integration/test_rl_metadrive_headless.py`
- Modify: `docs/phase4_implementation_log.md` (create if absent)

**Interfaces:**
- Verifies installed MetaDrive 0.4.3 and Gymnasium 1.3.0 behavior only.
- Does not add a scenario or alter MetaDrive termination rules.

- [x] **Step 1: Write the real `check_env` integration test**

```python
@pytest.mark.integration
def test_real_rl_environment_passes_gymnasium_checker() -> None:
    env = MultiAgentSpeedEnv(load_config("configs/base.yaml"))
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
```

- [x] **Step 2: Write real 100-step finite/deterministic tests**

Use seed 42 and a deterministic action sequence. Verify every Observation is `(24,)`, `float32`,
finite, and contained by the space; every Reward/component is finite; API is five elements; and close
runs. Run two short identical seed/action episodes and compare initial Observation and first 10 traces.

- [x] **Step 3: Run integration and observe the first concrete RED mismatch**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_rl_metadrive_headless.py -v`

Expected: fail on the first concrete MetaDrive/Gymnasium assumption until adapted.

- [x] **Step 4: Apply only the minimal API-compatible correction**

Inspect installed source before any change. MetaDrive 0.4.3 verified info keys are
`arrive_dest`, `crash_vehicle`, `crash_human`, `out_of_road`, and `max_step`.
Document the mismatch, cause, and correction in `docs/phase4_implementation_log.md`.

- [x] **Step 5: Re-run real integration and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_rl_metadrive_headless.py -v`

Expected: checker and real 100-step tests pass headlessly without a visible window.

- [x] **Step 6: Commit Task 5**

```powershell
git add tests/integration/test_rl_metadrive_headless.py docs/phase4_implementation_log.md src
git commit -m "test: verify real phase 4 gym environment"
```

---

### Task 6: PPO training, callbacks, checkpoint, and CLI

**Files:**
- Create: `src/mad_driving/training/__init__.py`
- Create: `src/mad_driving/training/callbacks.py`
- Create: `src/mad_driving/training/train.py`
- Create: `src/mad_driving/cli/train.py`
- Create: `tests/unit/training/test_callbacks.py`
- Create: `tests/unit/training/test_train.py`
- Create: `tests/unit/cli/test_train.py`

**Interfaces:**
- Produces `RewardComponentsCallback(BaseCallback)`.
- Produces immutable `TrainingResult(run_dir, final_checkpoint, best_checkpoint, timesteps)`.
- Produces `run_training(config, *, smoke, run_dir, resume_from=None, env_factory=...)`.
- CLI supports `--config`, `--smoke`, `--run-dir`, and `--resume-from`.

- [x] **Step 1: Write failing callback tests**

Inject SB3 callback locals containing vector `infos`; prove every known Reward key is recorded under
`reward/<key>` and missing/malformed info is ignored without breaking learning.

- [x] **Step 2: Write failing training orchestration tests with fake PPO and fake vector envs**

```python
def test_run_training_uses_only_configured_ppo_values_and_closes_envs(tmp_path) -> None:
    result = run_training(config, smoke=True, run_dir=tmp_path / "run", ...)
    assert fake_ppo.init_kwargs["policy"] == "MlpPolicy"
    assert fake_ppo.init_kwargs["n_steps"] == 2048
    assert fake_ppo.learn_kwargs["total_timesteps"] == 5_000
    assert fake_train_env.closed
    assert fake_eval_env.closed
    assert result.final_checkpoint == tmp_path / "run/checkpoints/final_model.zip"
    assert (tmp_path / "run/config_resolved.yaml").exists()
```

Add tests for 500,000 normal timesteps, unique train/eval env instances, checkpoint callback frequency
scaled by number of envs, resume using `PPO.load(..., env=...)`, `reset_num_timesteps=False`,
subprocess construction success, construction failure with verified cleanup and explicit error,
bounded worker teardown, and cleanup on `learn()`/save failure.

- [x] **Step 3: Write failing CLI tests**

Prove `--help`, required config, smoke forwarding, resume path validation, JSON success output, and
concise stderr/nonzero return without traceback.

- [x] **Step 4: Run Task 6 tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/training tests/unit/cli/test_train.py -v`

Expected: import failure because training modules and CLI do not exist.

- [x] **Step 5: Implement callback and training lifecycle**

```python
model = PPO(
    config.training.policy,
    train_env,
    learning_rate=config.training.learning_rate,
    n_steps=config.training.n_steps,
    batch_size=config.training.batch_size,
    n_epochs=config.training.n_epochs,
    gamma=config.training.gamma,
    gae_lambda=config.training.gae_lambda,
    clip_range=config.training.clip_range,
    ent_coef=config.training.ent_coef,
    vf_coef=config.training.vf_coef,
    max_grad_norm=config.training.max_grad_norm,
    seed=config.training.seed,
    tensorboard_log=str(tensorboard_dir),
    device="cpu",
)
```

Use built-in `CheckpointCallback` and `EvalCallback`; save `final_model.zip` explicitly. Serialize
`config.model_dump(mode="json")` to `config_resolved.yaml` with safe YAML and stable key order.
Use `SubprocVecEnv` when `num_envs > 1`; on construction failure, close partial resources, prove
workers stopped, and fail explicitly. Do not rebuild an equal-count `DummyVecEnv`: MetaDrive 0.4.3
allows only one engine per process, so that fallback is unsafe. For `num_envs == 1`, isolate the
separate evaluation environment in a one-worker `SubprocVecEnv` and use bounded, verified teardown.

- [x] **Step 6: Run Task 6 tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/training tests/unit/cli/test_train.py -v`

Expected: all callback, lifecycle, resume, process-isolation, serialization, and CLI tests pass.

- [x] **Step 7: Commit Task 6**

```powershell
git add src/mad_driving/training src/mad_driving/cli/train.py tests/unit/training tests/unit/cli/test_train.py
git commit -m "feat: train ppo coordinator with checkpoints"
```

---

### Task 7: Real checkpoint and TensorBoard integration

**Files:**
- Create: `tests/integration/test_ppo_checkpoint.py`
- Modify: `docs/phase4_implementation_log.md`

**Interfaces:**
- Exercises real Stable-Baselines3 2.9.0/PyTorch 2.8.0 with a tiny deterministic Gym env.
- Verifies project training orchestration without paying for a full MetaDrive smoke in normal tests.

- [x] **Step 1: Write a tiny real PPO integration test**

Use a deterministic 24D/Discrete(4) test env and a Phase 4 config with `n_steps=8`, `batch_size=8`,
`total_timesteps=16`, `checkpoint_interval_steps=8`, and `eval_interval_steps=8`.

Assert:

- periodic checkpoint zip exists;
- `best_model.zip` and `final_model.zip` exist;
- `PPO.load(final_checkpoint)` succeeds and predicts an integer action 0-3;
- TensorBoard contains at least one `events.out.tfevents.*` file;
- resolved config contains exact training values;
- resumed training creates a loadable final checkpoint.

- [x] **Step 2: Run the integration test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_ppo_checkpoint.py -v`

Expected: fail on the first real SB3 callback/path/signature mismatch until adapted.

- [x] **Step 3: Apply the smallest SB3 2.9.0 correction and document it**

Do not change PPO defaults. Record inspected constructor/callback/load signatures and the correction in
`docs/phase4_implementation_log.md`.

- [x] **Step 4: Re-run and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_ppo_checkpoint.py -v`

Expected: checkpoint, best model, final model, reload/predict, resume, and TensorBoard tests pass.

- [x] **Step 5: Commit Task 7**

```powershell
git add tests/integration/test_ppo_checkpoint.py docs/phase4_implementation_log.md src
git commit -m "test: verify ppo artifacts and resume"
```

---

### Task 8: Canonical 5,000-step smoke, quality gate, and publication

**Files:**
- Modify: `README.md`
- Modify: `docs/phase4_implementation_log.md`
- Modify: this plan checklist

**Interfaces:**
- Produces exact Windows/Linux training and resume commands.
- Produces a canonical real MetaDrive Phase 4 smoke record and a stacked Draft PR.

- [x] **Step 1: Run the complete quality gate**

```powershell
.venv\Scripts\python.exe -m pytest --cov=mad_driving --cov-report=term-missing -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
git diff --check feat/phase3-control-shield...HEAD
```

Expected: all tests pass, branch coverage is at least 80%, no lint/format/type/diff failures.

- [x] **Step 2: Run canonical real MetaDrive smoke training**

Run:

```powershell
.venv\Scripts\python.exe -m mad_driving.cli.train --config configs/train.yaml --smoke --run-dir runs/phase4_smoke_seed42
```

Expected: at least 5,000 timesteps complete headlessly on CPU; `best_model.zip`,
`final_model.zip`, resolved config, and TensorBoard event(s) exist. Periodic checkpoints are
required only when the configured 10,000-step interval is reached; their real SB3 save/load path is
covered by Task 7's shorter-interval integration test.

- [x] **Step 3: Load the produced checkpoint and run a deterministic prediction episode**

Use `PPO.load` with a fresh `MultiAgentSpeedEnv`, run until termination/truncation or 100 steps,
verify every Observation and Reward is finite, then close in `finally`.

- [x] **Step 4: Update README and implementation log**

Document setup with `--extra training`, smoke/standard/resume commands, output tree, exact dependency
versions, test/coverage evidence, real simulated/training steps, wall time, termination statistics,
checkpoint reload result, known upstream warnings, and any minimal API adjustments.

- [ ] **Step 5: Request broad Phase 4 code review and fix all Critical/Important findings**

Review the complete range from Phase 3 tip to Phase 4 HEAD for specification compliance, safety
fallbacks, lifecycle leaks, reward sign errors, observation leakage, Gymnasium semantics, seed
reproducibility, and SB3 artifact correctness. Re-run covering tests after each fix and re-review.

- [ ] **Step 6: Re-run final verification after review fixes**

Repeat Step 1 and the real Gym/checkpoint integration tests. Expected: same clean results.

- [x] **Step 7: Commit documentation and verification evidence**

```powershell
git add README.md docs/phase4_implementation_log.md docs/superpowers/plans/2026-07-20-phase4-rl-environment.md
git commit -m "docs: record phase 4 verification"
```

- [ ] **Step 8: Push and open a stacked Draft PR**

```powershell
git push -u origin feat/phase4-rl-environment
gh pr create --draft --base feat/phase3-control-shield --head feat/phase4-rl-environment
```

The PR body must state that it depends on #3, summarize the 24 features and ten Reward components,
show Gym checker/test/coverage evidence, report the real 5,000-step smoke and checkpoint reload, and
note that scenarios/evaluation remain Phase 5-6.

---

## Self-review record

- **Spec coverage:** MVP sections 7.2, 7.3, 7.4, 10, 11, Phase 4 of section 21, and Phase 4-relevant acceptance criteria map to Tasks 1-8.
- **Phase boundary:** specialized scenarios/curriculum remain Phase 5; evaluation artifacts, baselines, ablations, and visualization remain Phase 6.
- **Unspecified values:** normalization scales and Reward thresholds are explicit in the approved Phase 4 design and YAML, never hidden constants.
- **Collision model gap:** vehicle versus crossing-actor penalty is derived from verified MetaDrive `crash_vehicle`/`crash_human` step info; the fixed Snapshot schema is not expanded.
- **Type consistency:** `ObservationBuilder`, `RewardContext`, `RewardResult`, `RewardCalculator`, `MultiAgentSpeedEnv`, `TrainingResult`, and `run_training` signatures match producer/consumer tasks.
- **Placeholder scan:** no TBD/TODO/implement-later instruction remains; each task has exact files, behavior, commands, expected RED/GREEN results, and commit boundary.
