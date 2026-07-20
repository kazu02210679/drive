# Task 3 Report: ScenarioRuntime lifecycle boundary

## Summary

Implemented the standalone ScenarioRuntime lifecycle boundary without connecting it to
`MultiAgentSpeedEnv` (reserved for Task 8). The new surface provides immutable scene-frame
primitives, runtime state/result/context types, a lifecycle protocol, and a deterministic
no-op runtime.

## RED evidence

1. Command:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_runtime.py -q
   ```

   Result: exit code 1 during collection, as expected. The test failed because
   `mad_driving.interfaces` did not export `OcclusionRegion`.

2. After adding validation tests before implementation, command:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/unit/interfaces/test_models.py tests/unit/scenarios/test_runtime.py -q
   ```

   Result: exit code 1 during collection, as expected. `scene_frame` did not exist and
   `OcclusionRegion` was not exported. No production implementation existed at either RED.

## GREEN evidence

1. After the minimal implementation:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/unit/scenarios/test_runtime.py tests/unit/interfaces/test_models.py -q
   ```

   Result: 45 passed.

2. Required focused gates after the import-order refactor:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/unit/scenarios tests/unit/interfaces/test_models.py -q
   .venv\Scripts\ruff.exe check src/mad_driving/scenarios tests/unit/scenarios
   .venv\Scripts\mypy.exe src/mad_driving/scenarios src/mad_driving/interfaces/scene_frame.py
   ```

   Results: 50 passed; Ruff reported `All checks passed!`; mypy reported
   `Success: no issues found in 4 source files`.

## Full-suite evidence

Command:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Result: 447 passed, 19 warnings, in 25.61 seconds. The warnings are pre-existing third-party
Matplotlib deprecations and Stable-Baselines integration warnings; no test failures occurred.

## Files changed

- `src/mad_driving/interfaces/scene_frame.py` — frozen `OcclusionRegion` and `RoadContext`.
- `src/mad_driving/interfaces/__init__.py` — exports the scene-frame primitives.
- `src/mad_driving/scenarios/runtime.py` — lifecycle protocol, state/result/context types, and
  `NoOpScenarioRuntime`.
- `src/mad_driving/scenarios/__init__.py` — exports the runtime surface.
- `tests/unit/interfaces/test_models.py` — validates invalid occlusion boundaries and non-finite
  road conflict distances.
- `tests/unit/scenarios/test_runtime.py` — exercises lifecycle stability, fail-closed occlusion,
  visibility freezing, unique region IDs, and frozen scenario parameters.

## Self-review

- `OcclusionRegion` and `RoadContext` are frozen and reject empty IDs, malformed or non-finite
  boundaries, and non-finite optional conflict distances.
- `ScenarioState` takes a defensive parameter copy and exposes it through `MappingProxyType`.
- `ScenarioObservationContext` defensively converts actor IDs to `frozenset[str]`, rejects
  duplicate occlusion region IDs, and rejects active occlusion without visibility metadata.
- `NoOpScenarioRuntime` holds only its immutable scenario ID, returns fresh immutable state, and
  does not read or mutate the simulator in any lifecycle hook.
- No `MultiAgentSpeedEnv` changes or runtime integration were made.

## Concerns

No blocking concerns. The full test run emits unrelated dependency and integration warnings as
noted above. Runtime-to-environment wiring remains intentionally deferred to Task 8.
