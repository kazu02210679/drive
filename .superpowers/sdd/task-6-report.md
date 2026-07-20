# Task 6 Report: Aggregate Claims into Fixed Observation

## Scope and decisions

- Implemented only Task 6 coordinator and coordinator-test changes.
- Added frozen private `_AggregatedClaim` plus `aggregate_agent_claims(agent_id, claims)`.
- Aggregation validates every supplied claim, returns `None` only when the requested agent
  supplied no claim, and conservatively combines finite TTC/margin minima, severity and
  probability maxima, recommended-speed and confidence minima, and any hard-stop flag.
- `ObservationBuilder` keeps the existing 24 normalized slots, `float32` dtype, finite
  output, and `SceneObservation`-only boundary. Duplicate specialist claims are now valid.
- `RuleBasedCoordinator` continues to use every individual claim; a monotonicity regression
  test pins that adding a more hazardous claim cannot weaken its action.
- The observable predicted-hard-violation slot now uses only `RoadContext.stop_required` and
  `RoadContext.intersection_entry_prohibited`; collision and off-road are privileged state.

## TDD evidence

### RED

1. Added field-wise hazard aggregation, duplicate-agent 24D/dtype/finite, rule hard-stop,
   and rule-based safety-monotonicity tests before production edits.
2. Ran `.venv\\Scripts\\python.exe -m pytest tests/unit/coordinator -q`.
3. Result: **9 failed, 13 passed**. The new aggregation tests failed on the existing
   `duplicate agent_id` rejection. Existing coordinator tests also exposed the staged
   Task 4-to-Task 6 mismatch where production still accessed removed `SceneObservation`
   attributes (`previous_action`, collision, and off-road).
4. Corrected coordinator test fixtures to construct the immutable `SceneObservation` and
   `RoadContext` fields actually defined by Task 4, then re-ran RED. The same intended
   production gaps remained: removed-field accesses and duplicate-agent rejection.

### GREEN

- Implemented the minimal aggregation and `SceneObservation` migration.
- `.venv\\Scripts\\python.exe -m pytest tests/unit/coordinator -q`: **22 passed**
  (14 third-party Matplotlib/Pyparsing deprecation warnings).
- `.venv\\Scripts\\ruff.exe check src/mad_driving/coordinator tests/unit/coordinator`:
  passed.
- `.venv\\Scripts\\mypy.exe --follow-imports=skip src/mad_driving/coordinator`:
  passed, 3 source files checked.
- `.venv\\Scripts\\python.exe -m pytest tests/unit/agents tests/unit/coordinator tests/unit/safety -q`:
  **123 passed** (same 14 third-party warnings).
- `git diff --check`: passed.

## Required-check concern

The exact required command `.venv\\Scripts\\mypy.exe src/mad_driving/coordinator` was run
and is currently blocked by **18 pre-existing errors** in out-of-scope downstream Task 7/8
files: `src/mad_driving/envs/reward.py` and
`src/mad_driving/envs/multi_agent_speed_env.py`. They retain removed `SceneObservation`
fields and stale `AgentAnalysisResult` assumptions. The focused coordinator-only static
check above is clean.

The broader environment/reward command
`.venv\\Scripts\\python.exe -m pytest tests/unit/envs/test_multi_agent_speed_env.py tests/unit/envs/test_reward.py -q`
also cannot collect because `tests/unit/envs/test_reward.py` constructs the removed
`SceneObservation.off_road` field. That migration belongs to Task 7 and was intentionally
not changed.

## Review-fix: strict fixed-schema claim agent IDs

### Decision

- Added one shared Coordinator validator for the fixed specialist schema:
  `nominal`, `hazard`, and `rule` only.
- It rejects non-`str` values and unknown strings with `ValueError("invalid claim agent_id")`.
- `ObservationBuilder` runs the validator before all aggregation, including direct
  `aggregate_agent_claims` calls. `RuleBasedCoordinator` runs it before its defensive
  fallback/action selection.
- Absent valid specialists retain the existing finite observation slots and conservative
  missing-agent behavior.

### RED

- Added adversarial parameterized regressions in both observation and rule-based tests for
  `nan`, `inf`, `-inf`, `int`, `bool`, empty string, and unknown string.
- The observation regression includes a valid Nominal claim before the malformed unrelated
  claim, proving validation cannot be bypassed by aggregation filtering.
- `.venv\\Scripts\\python.exe -m pytest tests/unit/coordinator -q`: **14 failed,
  22 passed**. Non-empty malformed IDs were silently accepted; the empty string followed a
  generic defensive fallback instead of the fixed-schema error.

### GREEN

- `.venv\\Scripts\\python.exe -m pytest tests/unit/coordinator -q`: **36 passed**
  (14 third-party Matplotlib/Pyparsing deprecation warnings).
- `.venv\\Scripts\\ruff.exe check src/mad_driving/coordinator tests/unit/coordinator`:
  passed.
- `.venv\\Scripts\\mypy.exe --follow-imports=skip src/mad_driving/coordinator`:
  passed, 4 source files checked.
- `.venv\\Scripts\\python.exe -m pytest tests/unit/agents tests/unit/coordinator tests/unit/safety -q`:
  **137 passed** (same 14 third-party warnings).
