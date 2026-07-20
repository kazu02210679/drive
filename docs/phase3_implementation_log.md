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

MetaDrive's bundled `LaneChangePolicy.steering_control()` negates both the
wrapped lane-heading difference and lateral offset. The approved pseudocode
showed positive errors, which would use the opposite sign under the installed
coordinate convention. The custom Policy therefore follows MetaDrive 0.4.3:
positive lateral offset produces negative steering, and the heading error is
`vehicle_heading - lane_heading`, wrapped to `[-pi, pi]`.

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
