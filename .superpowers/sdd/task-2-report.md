# Phase 6 — Task 2 Report: Bind Profiles to Runtime Provenance

## Scope completed

- Bound `MultiAgentSpeedEnv` default suite composition and shield mode to the central method profile.
- Preserved injected `suite_factory` and `shield_factory` seams; an injected suite factory only replaces default suite construction.
- Added an immutable `MethodProfileSnapshot` to run metadata, raised the research contract to v7, and retained observation/action schema versions at v1.
- Validated the snapshot against resolved configuration during metadata construction/loading and against the active configuration during resume compatibility validation.
- Serialized the snapshot into fresh and resumed run metadata.

## Test-driven evidence

### RED

Command:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/envs/test_multi_agent_speed_env.py tests/unit/training/test_metadata.py tests/unit/training/test_train.py -q
```

Result: expected failure — `15 failed, 278 passed, 14 warnings in 12.67s`.

The failures demonstrated the intended missing behavior: every default environment used the legacy full suite, `config.shield.mode` overrode the central profile, metadata remained contract v6 without `method_profile`, and fresh training metadata omitted the snapshot.

### GREEN

Command:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/envs/test_multi_agent_speed_env.py tests/unit/training/test_metadata.py tests/unit/training/test_train.py -q
```

Result: `293 passed, 14 warnings in 11.98s`.

### Full suite

Command:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Result: `1014 passed, 36 warnings in 69.97s (0:01:09)`.

The warnings are existing third-party Matplotlib/PyParsing and Stable-Baselines3 warnings; no test failures remain.

## Files changed

- `src/mad_driving/envs/multi_agent_speed_env.py`
- `src/mad_driving/training/metadata.py`
- `src/mad_driving/training/train.py`
- `src/mad_driving/training/__init__.py`
- `tests/unit/envs/test_multi_agent_speed_env.py`
- `tests/unit/training/test_metadata.py`
- `tests/unit/training/test_train.py`
- `tests/integration/test_control_metadrive_headless.py`
- `tests/integration/test_rl_metadrive_headless.py`

## Self-review

- The environment resolves the central profile at construction; its default suite is built through `AgentSuite.from_config(..., method_id=...)`, while injected suite factories remain unchanged.
- Shield construction receives a copy of the resolved shield configuration with the profile’s default mode, so overlays cannot become a competing default-composition authority.
- The snapshot is frozen, canonicalized from the central registry, verified against `resolved_config.method.id`, required in serialized metadata, and checked again during resume contract validation.
- Existing integration coverage was updated from direct shield-overlay authority to the supported `proposed`/`proposed_no_shield` profile modes.

## Concerns

- None for Task 2 scope. The full suite retains pre-existing dependency warnings noted above.
