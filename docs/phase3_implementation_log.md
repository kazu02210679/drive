# Phase 3 implementation log

## Installed MetaDrive policy boundary

Inspected against the installed `metadrive-simulator==0.4.3` package before
adding production Policy code.

- `BasePolicy.__init__` is
  `(self, control_object, random_seed=None, config=None)`.
- `BasePolicy.reset(self)` clears `action_info`. A custom reset must call it
  before clearing PID state.
- `VehicleAgentManager` constructs the selected `agent_policy` with the vehicle
  object and generated seed. No custom constructor arguments are available.
- `BasePolicy.engine` resolves the active engine through `get_engine()`.
- `engine.external_actions` is assigned before Policy execution, and
  `EnvInputPolicy.act(agent_id)` reads `engine.external_actions[agent_id]`.
- A Policy class controls the environment action space through its classmethod
  `get_input_space()`. Returning `gymnasium.spaces.Discrete(4)` is compatible
  with the agent manager and does not depend on global configuration.

## Installed vehicle and lane units

- `vehicle.position` returns an x/y world position.
- `vehicle.heading_theta` is radians.
- `vehicle.speed` is m/s; `vehicle.speed_km_h` is km/h.
- `vehicle.max_speed_m_s` converts the configured `max_speed_km_h` to m/s.
- `vehicle.navigation.current_lane` is the primary lane reference, and
  `vehicle.lane` delegates to the same property in MetaDrive 0.4.3.
- Lane `local_coordinates(position)` returns longitudinal and lateral metres.
- Lane `heading_theta_at(longitudinal)` returns radians.
- Lane `speed_limit` is stored in km/h, so Policy code converts it with `/ 3.6`.
- The decision interval is `physics_world_step_size * decision_repeat`, using
  the existing validated `decision_interval_s` helper.

## Minimal compatibility choices

MetaDrive's bundled `LaneChangePolicy.steering_control()` passes negated errors
to its PID, but that bundled PID also negates its output. The first direct port
preserved the input negation while replacing the PID with positive-output
`BoundedPID`; the canonical control smoke then left the road at decision 32.
A real regression test reproduced the failure. Accounting for the bundled
PID's hidden output sign shows that the approved pseudocode was correct:
positive lateral offset produces positive steering, and heading error is
`lane_heading - vehicle_heading`, wrapped to `[-pi, pi]`. This minimal sign
correction keeps the installed coordinate and steering conventions intact.

`MetaDriveEnv.default_config()` returns a fresh `Config`. Its `update()` method
supports `allow_add_new_key=True`, so the smallest binding is a subclass whose
defaults add only `agent_policy` and the serialized `control_config`. A real
headless integration test will verify this closure-based subclass before the
binding is accepted.

The first headless reset showed one narrower compatibility difference:
MetaDrive recursively converts nested dictionaries under a new config key into
its own `Config` objects. Pydantic correctly rejected that object as a strict
`ControlConfig` input. MetaDrive's `Config.get_dict()` was verified to return a
plain nested dictionary, so Policy initialization normalizes through that
method before strict validation. No threshold or control value is changed.

The detailed plan's sample PID calculation allowed STOP to produce less than
full emergency braking at moderate speed, while the approved behavior and its
acceptance test require STOP to use emergency deceleration. STOP therefore
resets the longitudinal PID and directly commands the configured emergency
deceleration; the other three actions remain PID controlled.

## Verification environment

- Python 3.11.9
- metadrive-simulator 0.4.3
- Gymnasium 1.3.0
- Pydantic 2.11.7
- pytest 8.4.1
- Ruff 0.12.4
- mypy 1.16.1

## Canonical verification evidence

The canonical seed is 42 and one decision interval is 0.1 seconds.

- The shielded control smoke completed 100 decisions, or 10.0 simulated
  seconds, without termination or truncation.
- Its Action counts were `(78, 22, 0, 0)` for KEEP, SLOW, PREPARE_STOP, and
  STOP. The Shield made 0 actual interventions. The final Action was SLOW.
- The unchanged fixed-action smoke also completed 100 decisions, or 10.0
  simulated seconds, without termination or truncation and returned all three
  Agent claims.
- The real Policy integration covers `Discrete(4)`, finite steering and
  throttle/brake, forced STOP deceleration, and 60 KEEP decisions without an
  out-of-road result.
- The complete suite passed 267 tests with 97.02% branch coverage (reported as
  97% in the rounded table), above the required 80%.
- Strict mypy and Ruff checks pass for the Phase 3 boundaries. The final fresh
  repository-wide gate is recorded by the Phase 3 plan checklist.

The only accepted warnings are 14 upstream Pyparsing deprecations imported by
Matplotlib: one `oneOf`, six `parseString`, six `resetCache`, and one
`enablePackrat` warning. No project warning is suppressed.

## Pre-PR review hardening

The final read-only review found five boundary gaps. Phase 3 configuration now
uses a scalar-type-strict Pydantic base without changing the older YAML list to
tuple compatibility. Shield configuration enforces
`multiple_missing_action >= missing_agent_action`. Coordinator and Shield share
defensive dataclass reconstruction checks, with corrupted Coordinator input
returning PREPARE_STOP before arithmetic. The control smoke owns the environment
with `try/finally` immediately after construction, including component-factory
failures. A real 100-step complete-pipeline integration test makes the canonical
manual evidence an automated regression.
