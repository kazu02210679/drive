# Phase 2 Deterministic Agents Design

## Status and authority

This design implements Phase 2 of `docs/multi_agent_driving_mvp_spec.md`. That document remains the highest-level requirement. The user approved two narrow additions needed to make the specified Rule Agent executable:

- add `collision_occurred`, `off_road`, and `intersection_entry_prohibited` to `SceneSnapshot`;
- run all Phase 2 agents passively during the existing fixed-action smoke simulation and include the final claims and review in its JSON result.

The additions do not change the 4-action control scheme or the future 24-dimensional Coordinator observation.

## Scope

Phase 2 provides three deterministic claim-producing agents and one cross-review agent:

- `NominalMotionAgent`
- `HazardAgent`
- `RuleAgent`
- `CriticAgent`

Every decision step gives the same immutable `SceneSnapshot` to Nominal, Hazard, and Rule. Critic receives that snapshot and the resulting three `RiskClaim` values. The agents have no mutable episode state, random sampling, simulator access, model weights, or side effects.

Phase 2 does not implement action selection, longitudinal PID, lane keeping, Safety Shield, the Coordinator observation, PPO, scenario spawning, curriculum, JSONL traces, or comparison experiments. Agent output is observational: it cannot alter the fixed smoke action.

## Package structure

The new `src/mad_driving/agents/` package contains focused modules:

- `protocol.py` defines `DrivingAgent` with `agent_id: str` and `analyze(snapshot: SceneSnapshot) -> RiskClaim`.
- `kinematics.py` contains finite, side-effect-free projection, clearance, stopping-distance, stopping-margin, and safe-speed functions.
- `claim_factory.py` creates deterministic claim IDs and neutral claims without duplicating field defaults.
- `nominal.py`, `hazard.py`, and `rule.py` implement one claim-producing agent each.
- `critic.py` implements the eight required review rules without rerunning agents.
- `suite.py` calls the three agents in fixed order and then calls Critic once.
- `__init__.py` exposes only the public agents, suite, and protocol.

Configuration remains in `src/mad_driving/config/models.py`. The strict, frozen root model gains nested `NominalAgentConfig`, `HazardAgentConfig`, `RuleAgentConfig`, and `CriticAgentConfig` values under an `agents` key. Unknown keys, non-finite values, non-positive horizons, and invalid ranges stop startup.

## Snapshot boundary correction

`SceneSnapshot` gains three required booleans:

```python
collision_occurred: bool
off_road: bool
intersection_entry_prohibited: bool
```

`SceneSnapshotBuilder` reads collision and lane status from the MetaDrive ego object using the installed 0.4.3 API. The implementation will inspect those concrete properties before choosing names and will record any API correction in the Phase 2 implementation plan. Generic Phase 2 smoke has no ScenarioManager, so `intersection_entry_prohibited` is `False`. A keyword input on the builder lets the future ScenarioManager supply it without giving agents simulator access.

The same builder keyword boundary also accepts `stop_required`, `occlusion_present`, and `distance_to_conflict_point_m`; all keep their existing neutral defaults. This replaces Phase 1's hard-coded neutral scene flags while preserving identical default smoke behavior.

## Deterministic claim conventions

Agent IDs are fixed as `nominal`, `hazard`, and `rule`. Claim IDs use this exact format:

```text
{agent_id}:{step_index}:{target_actor_id-or-none}:{event_type}
```

No UUID or process-global counter is used. Agents inspect actors in `actor_id` order and use `actor_id` as the final tie-break. A claim is valid for the current decision step, so `valid_until_step == snapshot.step_index`.

Evidence and assumptions are stable code strings rather than free prose. This keeps equality checks and downstream logging reproducible. Every normal claim has non-empty evidence. When no actor or rule hazard exists, each agent still returns one neutral claim with `target_actor_id=None`, zero severity, and the ego speed limit as its recommended maximum speed.

## NominalMotionAgent

Nominal uses constant acceleration because the Phase 1 snapshot already carries actor acceleration and ego longitudinal acceleration. Its defaults are:

- horizon: `5.0 s`
- time step: `0.25 s`
- ego length and width: `4.5 m`, `1.8 m`
- lane half-width for cut-in candidacy: `1.75 m`
- longitudinal and lateral clearance buffers: `0.5 m`, `0.25 m`
- probability TTC scale: `3.0 s`
- probability distance scale: `3.0 m`

At each time `t` from 0.25 through 5.0 seconds, the ego-fixed-frame relative position is:

```text
r(t) = r_actor(0)
     + (v_actor_ego_frame - [ego_speed, 0]) * t
     + 0.5 * (a_actor_ego_frame - [ego_acceleration, 0]) * t²
```

The collision envelope is the sum of half vehicle dimensions plus the configured buffers. Clearance is the Euclidean norm of the positive distance outside that rectangular envelope. TTC is the earliest sampled time at which both longitudinal and lateral envelope distances are non-positive. The minimum sampled clearance is retained even when no collision is predicted.

The evaluated actor set contains:

- same-lane actors ahead of the ego;
- non-same-lane actors whose predicted lateral position enters the configured lane half-width;
- actors with `actor_type == "crossing_actor"`.

For each candidate:

```text
ttc_term      = exp(-ttc / 3.0) when TTC exists, otherwise 0
distance_term = exp(-minimum_clearance / 3.0)
closing_term  = clip(longitudinal_closing_speed / 10.0, 0, 1)
probability   = clip(0.65*ttc_term + 0.25*distance_term + 0.10*closing_term, 0, 1)
severity      = probability
recommended_max_speed = speed_limit * (1 - 0.75*severity)
```

The selected claim is the highest tuple `(severity, probability, -minimum_clearance)` with `actor_id` as the ascending tie-break. It reports `min_ttc_s`, but leaves `stopping_margin_m=None` because stopping margin belongs to Hazard.

## HazardAgent

Hazard evaluates physically conservative alternatives without copying the future Safety Shield. Its defaults are:

- lead maximum deceleration: `-8.0 m/s²`
- crossing actor maximum speed: `8.0 m/s`
- ego reaction delay: `0.5 s`
- ego maximum safe deceleration: `-6.0 m/s²`
- ego length: `4.5 m`
- safety buffer: `2.0 m`
- occlusion crawl speed: `2.0 m/s`
- severe negative-margin scale: `10.0 m`
- crossing occupancy allowance: `1.0 s`

For a same-lane actor ahead, longitudinal velocity is projected into the ego heading. With positive deceleration magnitudes `b_lead=8` and `b_ego=6`:

```text
bumper_gap = actor_longitudinal
           - 0.5*(ego_length + actor_length)
available_distance = bumper_gap
                   + lead_speed²/(2*b_lead)
                   - safety_buffer
required_distance = ego_speed*reaction_delay
                  + ego_speed²/(2*b_ego)
stopping_margin = available_distance - required_distance
```

The maximum speed that can stop inside a non-negative available distance is the positive solution of `d = v*t_reaction + v²/(2*b_ego)`:

```text
safe_speed = max(0, sqrt((b_ego*t_reaction)² + 2*b_ego*d)
                    - b_ego*t_reaction)
```

Severity is the smooth function `1 / (1 + exp(stopping_margin / 10.0))`. It is `0.5` at zero margin, greater than `0.5` for negative margin, and approaches zero as positive margin grows. Hazard remains advisory and never turns this score directly into a hard stop.

For a crossing actor or an occluded conflict point, available distance is `distance_to_conflict_point_m - safety_buffer` and ego arrival time is conflict distance divided by ego speed. A visible crossing actor's effective worst-case speed is the greater of its observed lateral speed magnitude and the configured `8.0 m/s` maximum; this prevents an already faster observed actor from being artificially slowed by configuration. Its earliest arrival is absolute lateral distance divided by that effective speed. It is a conflict candidate when earliest actor arrival is no later than ego arrival plus the configured `1.0 s` occupancy allowance. Because Hazard evaluates the worst case, the actor may delay entry to coincide with ego arrival; the resulting worst-case TTC is ego arrival time. An occluded virtual actor is assumed able to synchronize immediately, so it uses the same ego-arrival TTC. The stopping equation determines margin and recommended speed. If occlusion exists but conflict distance is unavailable, the agent cannot invent geometry: it returns `stopping_margin_m=None`, severity `0.7`, and the configured `2.0 m/s` crawl recommendation while recording the missing-distance assumption.

Hazard selects the candidate with highest severity, then lower finite TTC, then ascending target actor ID. A missing TTC sorts after every finite TTC. A neutral no-hazard claim is returned only when no lead, crossing, or occlusion candidate exists. Hazard never sets `hard_stop_required`; that remains Rule and later Safety Shield responsibility.

## RuleAgent

Rule is a deterministic constraint module with no prediction model. Its base recommendation is `ego.speed_limit_mps`.

The following conditions require `hard_stop_required=True` and `recommended_max_speed_mps=0.0`, in this priority order:

1. `collision_occurred`
2. `off_road`
3. `intersection_entry_prohibited`
4. `stop_required`

If none applies, Rule returns a speed-limit claim. Its severity is `clip((ego.speed_mps - speed_limit) / max(speed_limit, 1.0), 0, 1)` and its probability is `1.0` when currently over the limit, otherwise `0.0`. A zero speed limit is valid and recommends zero without inventing a hard stop unless one of the four explicit conditions is true.

## CriticAgent

Critic runs the required checks once, in the fixed order below. Reason codes are stable and duplicate-free:

1. `nominal_hazard_disagreement`: Nominal severity is below `0.3` and Hazard stopping margin is negative.
2. `occlusion_underestimated`: occlusion exists and Nominal severity is below `0.3`.
3. `hard_stop_conflict`: Rule requests hard stop while another agent recommends more than `0.0 m/s`.
4. `speed_recommendation_spread`: maximum minus minimum recommendation exceeds `5.0 m/s`.
5. `claim_expired`: `valid_until_step < snapshot.step_index`.
6. `low_confidence_definitive`: confidence is below `0.5` and the claim either requests hard stop or recommends at most half the speed limit.
7. `missing_evidence`: evidence is empty.
8. `invalid_claim`: a field is non-finite or outside the RiskClaim ranges.

The normal `RiskClaim` constructor prevents invalid values. Critic still validates defensively because claims may come from deserialization or a future external component. Invalid claims are challenged, never used in arithmetic, and force `max_severity=1.0`.

`challenged_claim_ids` follows input claim order with duplicates removed. `supported_agent_ids` contains valid agents whose claims triggered no challenge, sorted by agent ID. `reasons` follows the rule order above. `conflict_score` is the number of distinct triggered rules divided by eight and clipped to `[0, 1]`. `unresolved_conflict` is true when at least one rule triggered. Otherwise, `max_severity` is the maximum valid claim severity or zero for an empty claim sequence.

## AgentSuite and smoke integration

`AgentSuite.analyze(snapshot)` invokes Nominal, Hazard, and Rule exactly once in that order, then invokes `CriticAgent.review(snapshot, claims)` exactly once. It returns `(claims, review)` without retaining them.

`SmokeResult` gains:

```python
final_claims: tuple[RiskClaim, ...]
final_review: CriticReview
```

The smoke loop builds one snapshot after each MetaDrive step and immediately runs the suite. Only the final snapshot, three claims, and review are emitted in the existing JSON result. This bounds console output while proving that every decision step can be analyzed. The configured fixed action remains unchanged.

## Configuration

`configs/base.yaml` gains an `agents` section with all defaults written explicitly. Pydantic constraints include:

- horizons, time steps, dimensions, buffers, scales, speeds, and thresholds that represent magnitudes are finite and positive, except safety buffer may be zero;
- physical decelerations are finite and strictly negative in YAML;
- probabilities and confidence thresholds are in `[0, 1]`;
- Nominal time step cannot exceed its horizon;
- Critic speed-spread threshold is non-negative.

The agent classes receive their frozen config objects at construction. They do not read global configuration or environment variables.

## Error handling

Snapshot and claim dataclasses continue rejecting non-finite or out-of-range values at construction. Kinematic helpers reject non-finite inputs and non-positive braking magnitudes rather than silently clipping invalid physics. Valid physical edge cases—zero speed, zero closing speed, no actors, no conflict point, and non-negative stopping margin—return finite neutral results.

The smoke CLI retains its concise stderr error and nonzero exit behavior for invalid configuration or simulator failure. Environment closure remains protected by `finally`, including agent-analysis failures.

## Test strategy

Tests follow red-green-refactor for each unit:

- config tests cover defaults, explicit values, unknown keys, finite checks, sign constraints, and cross-field horizon/time-step validation;
- snapshot tests cover the three new booleans, builder keyword scene flags, and real MetaDrive collision/off-road accessor compatibility;
- kinematics tests cover zero speed, zero closing speed, envelope overlap, negative gaps, stopping margin signs, and safe-speed inversion;
- Nominal tests cover same-lane front actors, cut-in candidates, crossing actors, no actors, stable tie-breaking, five-second sampling, and identical repeated output;
- Hazard tests cover lead maximum braking, positive and negative margins, crossing timing, occluded virtual actors, missing conflict distance, and deterministic output;
- Rule tests cover speed limit and each hard-stop condition independently, including priority;
- Critic tests cover all eight required rules independently and in combination, stable order, invalid injected data, and empty claims;
- suite tests prove fixed call order and one review pass without simulator coupling;
- smoke unit and integration tests prove three final claims plus one review, unchanged action application, finite JSON serialization, and guaranteed close;
- the full quality gate requires all tests, at least 80% branch coverage, Ruff lint and format, mypy strict, and a real 100-step headless MetaDrive smoke run.

Upstream Matplotlib/pyparsing deprecation warnings already documented in Phase 1 are not treated as project-code failures.

## Git and delivery

Work occurs on `feat/phase2-deterministic-agents`, based on the reviewed Phase 1 commit `e826c95`. Phase 1 PR #1 remains unchanged. After verification, Phase 2 is pushed as a separate branch and opened as a draft pull request targeting `feat/phase1-foundation`. Once Phase 1 merges, the Phase 2 PR can be retargeted to `main` without combining implementation histories.

## Acceptance conditions

Phase 2 is complete when all of the following are true:

- all three claim-producing agents and Critic implement the approved deterministic behavior;
- each smoke decision step can produce three claims and one review without changing vehicle control;
- the final smoke JSON contains the final claims and review;
- repeated analysis of the same snapshot and settings is exactly equal, including claim IDs and ordering;
- all Phase 1 tests remain green;
- unit and real MetaDrive integration tests pass with at least 80% branch coverage;
- Ruff, formatting, and mypy strict checks pass;
- the 100-step headless smoke command exits zero;
- API mismatches and only their smallest corrections are recorded before code is changed.
