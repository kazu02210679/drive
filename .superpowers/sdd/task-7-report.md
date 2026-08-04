# Task 7 Report: Current-state reward and privileged outcome inputs

## Scope

Implemented only Phase 4.1 Task 7 from base `a2b1a8b`.

## Changes

- Replaced the snapshot/claim reward input with `RewardContext(previous_frame, next_frame, analysis, executed_action, shield_intervened, decision_interval_s)`.
- Read arrival, collision, and off-road labels only from `next_frame.privileged`.
- Kept transition calculations on frame observations: progress uses previous-heading movement, and jerk/standstill use the post-step observation.
- Removed the lookahead config and `_safe_brake_streak`; safe braking is judged immediately from current analysis and post-step privileged state.
- Treated missing TTC as safe, required present non-failed Hazard and Rule claims, and retained the one-time/resettable arrival guard.
- Removed the retired YAML setting from both active configs; strict model validation now rejects it.

## TDD evidence

RED, before production edits:

```text
.venv\Scripts\python.exe -m pytest tests/unit/envs/test_reward.py tests/unit/config/test_rl_config.py -q
40 failed, 18 passed
```

The failures were the missing frame-based `RewardContext` fields and acceptance of `unnecessary_brake_lookahead_steps`.

GREEN after the minimal implementation:

```text
.venv\Scripts\python.exe -m pytest tests/unit/envs/test_reward.py tests/unit/config/test_rl_config.py -q
58 passed, 14 warnings
```

## Verification

- `pytest tests/unit/config -q`: 66 passed.
- `ruff check src/mad_driving/envs/reward.py tests/unit/envs/test_reward.py tests/unit/config/test_rl_config.py src/mad_driving/config/models.py`: passed.
- Both `configs/base.yaml` and `configs/train.yaml` load through the source-tree configuration loader.
- `git diff --check`: passed.

## Staged dependency concern

The broad `tests/unit/envs -q` suite remains red before reward execution because its Task 8-owned test harness still constructs the removed `SceneObservation` fields (`seed`, `previous_action`, and outcome labels). The exact `mypy src/mad_driving/envs/reward.py` command follows package imports and likewise reports 19 pre-existing/staged `multi_agent_speed_env.py` migrations, including the intentionally deferred `RewardContext` call-site conversion. Task 8 owns the frame lifecycle and environment integration; no compatibility fields were restored to agent-visible observations.
