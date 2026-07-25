# Phase 6 Evaluation Design

**Status:** approved for implementation on 2026-07-23

**Branch:** `feat/phase6-evaluation`

**Base:** `feat/phase5-scenarios` at `a95c7db`

**Top-level requirement:** `docs/multi_agent_driving_mvp_spec.md`

## Goal

Build the reproducible Phase 6 evaluation pipeline for the MVP. The pipeline must run
the defined baselines and ablations on matched scenario seeds, preserve strict
provenance, calculate the required safety, efficiency, comfort, and multi-Agent
metrics, and generate JSONL, CSV, PNG, GIF, and Markdown artifacts.

This branch implements and smoke-tests the evaluation infrastructure. The expensive
formal experiment over all methods and policy seeds `42, 43, 44, 45, 46` is a separate
execution gate after the infrastructure and its PR pass review.

## Fixed Scope

Phase 6 includes:

- immutable method profiles for B0, B1, B2, Proposed, and the three specified
  ablations;
- method-bound training and checkpoint provenance for PPO-based profiles;
- fixed all-level validation for final checkpoint selection;
- final evaluation on the isolated test split;
- strict per-step and per-episode evaluation artifacts;
- matched-seed aggregation and comparison;
- required plots, representative GIF overlays, and a Markdown comparison report;
- CLI entry points for single-method evaluation, comparison, and offline rendering;
- unit tests, deterministic fake-environment integration tests, and a short real
  MetaDrive smoke comparison;
- README commands and an implementation log;
- a follow-up update to the separate `feat-architecture-map-flow-cards` branch after
  the Phase 6 implementation is complete.

Phase 6 does not include:

- the full five-policy-seed benchmark execution;
- changes to the 24-dimensional observation schema;
- the deferred `ttc_valid`, `claim_valid`, `agent_failed`, or
  `target_actor_present` observation slots;
- image perception, learned steering, lane changes, LLM/VLM Agents, or real-car
  integration;
- a notebook-only or manually edited evaluation workflow.

## Design Choice

The implementation uses a staged artifact pipeline:

```text
method profile + checkpoint + explicit episode plan
                         |
                         v
                 evaluation runner
                         |
              strict step JSONL artifacts
                         |
                         v
                episode metric reducer
                         |
                episode_metrics.csv
                         |
                         v
             matched comparison validator
                         |
                 comparison.csv
                 /      |      \
                v       v       v
              plots    GIFs   Markdown report
```

Simulation is the only stage allowed to create raw episode data. Metrics,
comparisons, plots, GIF overlays, and reports consume published artifacts and can be
regenerated without another simulation run. This makes expensive experiments
restartable and prevents plotting or report code from silently changing simulation
semantics.

The rejected alternatives are:

- a monolithic command that simulates, aggregates, and renders in one process, because
  partial failure would make safe resume and independent artifact validation difficult;
- notebook-driven analysis, because it does not provide a CI-verifiable, immutable,
  repeatable research contract.

## Comparison Methods

`MethodId` is a closed string enum. A central registry is the only source of Agent,
Critic, policy, and default training-Shield composition. YAML selects a method by ID;
it cannot provide a second, conflicting list of enabled components.

| Method ID | Specialists | Critic | Action source | Default training Shield |
|---|---|---|---|---|
| `b0_rule` | none | no-op | deterministic visible-scene TTC rule | `enforce` |
| `b1_nominal` | Nominal | no-op | PPO | `enforce` |
| `b2_multi_no_review` | Nominal, Hazard, Rule | no-op | PPO | `enforce` |
| `proposed` | Nominal, Hazard, Rule | enabled | PPO | `enforce` |
| `proposed_no_critic` | Nominal, Hazard, Rule | no-op | PPO | `enforce` |
| `proposed_no_shield` | Nominal, Hazard, Rule | enabled | PPO | `off` |
| `proposed_no_hazard` | Nominal, Rule | enabled | PPO | `enforce` |

The no-op Critic returns a validated neutral `CriticReview`: zero conflict and
severity, no unresolved conflict, no supported Agents, no challenged Claims, and a
single diagnostic reason identifying intentional review disablement. It is not a
runtime Critic failure.

Intentional specialist omission is represented by `None` in `AgentSuite` and by the
resulting `expected_agent_ids`. It is never added to `failed_agent_ids`, so the
ablation-aware Shield does not treat the configuration as a runtime failure.

`AppConfig` gains a strict `method` section with only `id: MethodId`. `configs/base.yaml`
defaults to `proposed`. Method-specific overlays live below `configs/methods/`. Training
metadata and evaluation manifests record the method ID and resolved central profile.
Changing the method is checkpoint- and resume-incompatible.

This provenance change raises the research contract from v6 to v7. Observation schema
v1, shape `(24,)`, action schema v1, and the four-action order remain unchanged.
Evaluation step, episode-summary, evaluation-manifest, and selected-checkpoint
artifacts begin at their own schema version 1.

### B0 rule

B0 uses only the Agent-visible `SceneObservation`; it cannot read
`PrivilegedWorldState`, specialist Claims, or a PPO observation. The environment
exposes its current immutable `SceneObservation` through an evaluation-only read
method. The rule selects:

1. `STOP` for a visible stop requirement or prohibited intersection entry;
2. `STOP` when the minimum visible constant-velocity TTC is at most `1.0` seconds;
3. `PREPARE_STOP` when it is at most `3.0` seconds;
4. `SLOW` when it is at most `5.0` seconds;
5. `KEEP` otherwise.

Only visible actors in the ego lane or on an intersecting trajectory are eligible.
The TTC calculation reuses the validated kinematic coordinate conventions and never
uses hidden scenario truth.

## Comparison Tracks

Three explicit tracks prevent Shield effects from being mixed with raw decision
quality:

| Track | Methods | Evaluation Shield mode |
|---|---|---|
| `decision` | B1, B2, Proposed | `monitor` for every method |
| `system` | B0, B1, B2, Proposed | `enforce` for every method |
| `ablation` | Proposed and all three ablations | `enforce`, except `proposed_no_shield=off` |

The comparison-plan validator rejects a method set or Shield mode that violates the
selected track. `proposed_no_shield` is an ablation and cannot be labeled as a main
baseline.

Training and evaluation are distinct. PPO methods must be trained with their own
method profile. An evaluation runner refuses a checkpoint whose recorded method ID,
observation schema, action schema, research contract, or resolved config hash does not
match the requested evaluation input. B0 never accepts a checkpoint.

## Scenario and Seed Contract

The fixed all-level matrix is:

| Case ID | Level | Scenario |
|---|---:|---|
| `level0_nominal` | 0 | `nominal` |
| `level1_lead_brake` | 1 | `lead_brake` |
| `level2_lead_brake` | 2 | `lead_brake` |
| `level2_cut_in` | 2 | `cut_in` |
| `level3_occluded_crossing` | 3 | `occluded_crossing` |

An `EvaluationPlan` contains an ordered tuple of case IDs and explicit non-negative
episode RNG seeds. Smoke mode uses one episode per case. Formal plans may use more but
must keep the same ordered `(case_id, episode_rng_seed)` matrix for every compared
method.

The existing role-scoped allocator remains authoritative:

- train: MetaDrive scenario identities in `[0, 10000)`;
- validation: `[10000, 11000)`;
- test: `[20000, 21000)`.

Validation is used for checkpoint selection and never for final claims. Test is used
only after checkpoint selection and never for curriculum, training, hyperparameter
choice, or model selection. Typed APIs use `Literal["validation", "test"]`, and the
model selector accepts only validation artifacts.

The scenario runtime receives an explicit finite evaluation schedule for both
validation and test roles. Training environments reject such a schedule. Every
episode records the episode RNG seed, actual MetaDrive scenario index, independent
scenario-selection seed, scenario-parameter seed, case ID, scenario ID, level, role,
method, track, and policy seed.

## Fixed All-Level Checkpoint Selection

PPO candidate checkpoints are discovered only from a completed training run's
verified metadata inventory. The candidate set may include periodic, level-best, and
final checkpoints, but every ZIP must have a valid bound curriculum sidecar and
recorded hash.

Every candidate is evaluated on the same ordered all-level validation plan. Selection
uses this deterministic ordering:

1. highest mean episode reward;
2. lowest collision rate;
3. highest success rate;
4. highest mean route completion;
5. lowest training timestep, preferring the earlier checkpoint on an exact tie;
6. lexicographically smallest checkpoint SHA-256 as a final deterministic tie-break.

The result is written to `model_selection.csv` plus a strict selected-checkpoint
record containing path, hash, method, policy seed, and validation-plan hash. Test
evaluation requires this record for PPO methods and refuses an arbitrary unselected
checkpoint unless an explicit `--smoke` flag is active.

## Runtime Evaluation Interfaces

The core interfaces are small and simulator-independent:

- `MethodProfile`: immutable method composition and policy kind;
- `EvaluationCase`: one level/scenario pair;
- `EvaluationEpisodeKey`: method, track, role, policy seed, case, and episode RNG seed;
- `EvaluationPolicy`: returns one `DrivingAction` from the current PPO observation and
  Agent-visible scene;
- `EvaluationStepRecord`: one strict post-transition record;
- `EpisodeMetrics`: one immutable reduced row;
- `EvaluationManifest`: resolved provenance and hashes for a published evaluation;
- `ComparisonRow`: one aggregate metric row for one method/track/scenario grouping.

PPO and B0 are adapters behind `EvaluationPolicy`. PPO ignores the scene argument and
uses deterministic `model.predict`. B0 ignores the 24-vector and uses only the visible
scene. Evaluation always runs one MetaDrive environment at a time in Phase 6; no new
parallel simulator topology is introduced.

`policy_seed` is a required integer for PPO methods and `None` for B0. B0 runs each
physical episode once; it is not duplicated five times and treated as five independent
policies. All methods must share the same `(case_id, episode_rng_seed)` physical plan.
PPO methods must additionally share the same policy-seed set with one selected
checkpoint per policy seed.

## Step Record Schema

Each JSONL line contains exactly one post-transition `EvaluationStepRecord` with:

- schema version and episode key;
- step index, simulation time, and decision interval;
- episode and scenario seed identity;
- requested, Shield-required, and executed actions;
- raw unsafe-request and actual Shield-intervention booleans and reasons;
- target speed, ego speed, longitudinal acceleration, route progress, and lane offset;
- method-independent oracle collision state and collision kind;
- method-independent oracle minimum actual TTC;
- method-independent oracle minimum stopping margin;
- pre-step hard-rule constraint and post-step rule-violation event;
- scenario success, scenario failure, arrival, off-road, termination, and truncation;
- cumulative unnecessary-stop duration;
- reward total and all ten reward components;
- all RiskClaims, CriticReview, expected Agent IDs, failed Agent IDs, and sanitized
  errors;
- policy-inference, Agent-analysis, Shield, and total-decision latency in milliseconds;
- optional relative frame path for a captured representative episode.

`PrivilegedWorldState` gains `minimum_actual_stopping_margin_m: float | None`. It is
computed by the fixed oracle path, is available only to reward/evaluation/debug code,
and is not added to the Coordinator observation. Negative stopping-margin metrics
must never be calculated from a comparison Agent's Claim.

For every actor with a finite fixed-oracle rectangle-entry TTC `t`, the oracle margin
is:

```text
available_distance = ego_speed * t
required_distance = ego_speed * oracle_reaction_delay
                    + stopping_distance(ego_speed, oracle_safe_deceleration)
stopping_margin = available_distance - required_distance
```

The state stores the minimum finite actor margin, or `None` when no actor has a finite
rectangle-entry TTC. `oracle_reaction_delay` and `oracle_safe_deceleration` are copied
from the resolved Hazard configuration even when the Hazard Agent is intentionally
disabled. The comparator requires these fixed oracle values to match across methods.
They are evaluation physics constants, not Agent output.

Latency is descriptive wall-clock telemetry and is excluded from deterministic
artifact equality checks. Every latency must still be finite and non-negative.

JSON uses UTF-8, sorted keys, compact separators, no NaN/Infinity, and one trailing
newline per record. Readers reject unknown or missing fields, duplicate JSON keys,
non-finite values, malformed enum values, and non-contiguous step indices.

## Episode Metrics

The reducer calculates one strict row per episode.

### Safety

- collision and crossing-actor collision indicators;
- near-miss indicator and minimum actual TTC;
- negative oracle stopping-margin indicator and minimum oracle stopping margin;
- hard-rule violation indicator;
- raw unsafe-request rate;
- Shield intervention rate;
- off-road indicator.

A near miss is a non-collision episode with at least one finite oracle TTC below the
configured `reward.near_miss_ttc_s`. A rule violation occurs when the pre-step oracle
requires a hard stop and the executed action is below `STOP`.

### Efficiency

- scenario success;
- final route completion;
- time-weighted average speed;
- simulated travel time;
- unnecessary-braking event count;
- cumulative unnecessary-stop duration.

An unnecessary-braking event starts when the existing oracle-based unnecessary-brake
condition changes from false to true; consecutive braking steps count once.

### Comfort

- time-weighted longitudinal-acceleration RMS;
- maximum deceleration magnitude;
- time-weighted jerk RMS using adjacent acceleration records and simulation time.

### Multi-Agent and runtime

- Agent disagreement eligible steps, count, and rate;
- Critic challenge eligible steps, count, and rate;
- Critic-found-missed-danger count and rate;
- Critic false-challenge count and rate;
- Agent-failure fallback count;
- decision-latency p50, p95, and p99.

Agent disagreement means that at least two present specialist aggregates map to
different discrete actions. A Critic-found missed danger is a challenged Nominal Claim
on a step where a supported Hazard or Rule Claim maps to a stricter action. A false
challenge is a challenged Claim on a step with no hard-rule constraint, no collision,
non-negative oracle stopping margin, and no oracle TTC below the near-miss threshold.

Rates with no eligible denominator are serialized as an empty CSV cell and rendered
as `N/A`, not as zero. This distinction matters for B0 and B1.

## Matched Comparison

Before aggregation, the comparator verifies:

- one and only one row for every expected episode key;
- identical `(track, case_id, episode_rng_seed)` physical plans across all methods;
- identical policy-seed sets across PPO methods, while B0 has `policy_seed=None` and
  exactly one row per physical episode;
- identical derived scenario seeds and sampled scenario parameters for matched rows;
- one shared observation/action/research schema contract;
- required Shield mode per track;
- no training or validation role rows in a final test comparison;
- no test role rows in model selection;
- no failed or incomplete evaluation manifest.

Any mismatch aborts the comparison. The comparator does not drop unmatched rows or
silently impute values.

`comparison.csv` contains method, track, scenario grouping, physical episode count,
policy-replicate count, mean, and sample standard deviation across independent
policy-seed means where defined. B0 and smoke output with one PPO seed leave the
policy-replicate standard deviation blank. Smoke output is labeled
`SMOKE — NOT A RESEARCH RESULT`.

## Artifact Layout and Publication

```text
evaluations/<evaluation_id>/
├─ config_resolved.yaml
├─ evaluation_plan.yaml
├─ evaluation_manifest.json
├─ model_selection.csv                 # validation runs only
├─ selected_checkpoints.json           # one record per PPO method/policy seed
├─ episodes/
│  └─ <method>/<track>/<policy_seed-or-rule>/<case_id>/
│     ├─ episode_<seed>_trace.jsonl
│     ├─ episode_<seed>_summary.json
│     └─ episode_<seed>_frames/         # only when explicitly captured
├─ metrics/
│  ├─ train_metrics.csv
│  ├─ eval_metrics.csv
│  └─ comparison.csv
├─ plots/
│  ├─ learning_curve.png
│  ├─ collision_rate.png
│  ├─ success_route_completion.png
│  ├─ unnecessary_braking.png
│  ├─ comfort.png
│  └─ agent_disagreement.png
├─ renders/
│  └─ <method>_<policy_seed-or-rule>_<case_id>_<seed>.gif
└─ comparison_report.md
```

The writer reserves an empty destination and builds all outputs in a private sibling
staging directory. Files are flushed and closed before hashing. The manifest is
written last and inventories every published artifact by relative path, byte size,
and SHA-256. Publication uses one directory rename. Existing or non-empty destinations
are never merged or overwritten. On failure, the uncommitted staging directory is
removed only after its resolved path is proven to be inside the reserved parent.

Readers validate the manifest and all hashes before consuming artifacts. Plotting,
rendering, and reporting refuse a failed integrity check.

## Visualization and Reporting

Matplotlib uses a non-interactive backend and fixed dimensions, labels, method order,
colors, and DPI. Plots are generated only from validated CSV rows. Missing rates are
shown as `N/A`; smoke figures include a visible smoke watermark.

Representative GIFs are opt-in and are not generated for every formal episode. A
captured RGB frame is stored losslessly before GIF encoding. The overlay shows method,
track, scenario, seed, step, requested/required/executed action, speed, oracle TTC,
speed limit, target speed, Shield status, each present Agent's severity and recommended
maximum speed, and Critic conflict score. GIF generation consumes saved frames plus
strict JSONL records and never reruns MetaDrive.

MetaDrive 0.4.3 compatibility note: `image_observation=True` changes MetaDrive's
observation construction and requires an explicitly registered image sensor. On this
Windows workspace, its 3D asset-path conversion also cannot open assets below a
non-ASCII user path. Phase 6 therefore captures visualization-only RGB frames through
MetaDrive's standard headless top-down renderer (`window=False`). This adapter does
not enable image observation, does not feed pixels to any policy, and leaves the
24-dimensional PPO observation unchanged.

The Markdown report is deterministic and contains:

- provenance and the exact method/seed/scenario matrix;
- selected checkpoint hashes;
- separate decision, system, and ablation tables;
- safety, efficiency, comfort, multi-Agent, and latency metrics;
- links to plots and GIFs using relative paths;
- explicit `N/A` explanations;
- a smoke/non-formal warning when applicable;
- failures or exclusions only when they are represented in the manifest, never as
  manually edited prose.

## Training Metrics Extraction

Phase 6 adds an offline TensorBoard extractor. It reads event files only from a
verified completed training run, hashes each input event file into the evaluation
manifest, and writes long-form `train_metrics.csv` rows with method, policy seed,
training timestep, tag, and finite scalar value.

The required exported series are episode reward, every reward component, policy
entropy, value loss, and explained variance. Stable-Baselines3's finite
`train/entropy_loss` is exported as `policy_entropy = -entropy_loss`; the original tag
and scalar are retained for provenance. Missing required tags fail a formal report but
may be explicitly listed as unavailable in smoke mode. `learning_curve.png` and reward
component trend sections are generated from this CSV, never by reading TensorBoard
during plotting or report generation.

`eval_metrics.csv` is the canonical one-row-per-episode output of the episode reducer.
It replaces the internal phrase `episode_metrics.csv`; comparison and visualization
consume only this canonical filename.

## CLI

The user-facing commands are:

```powershell
python -m mad_driving.cli.evaluate --plan configs/evaluation/phase6_smoke.yaml --output evaluations/phase6_smoke --smoke
python -m mad_driving.cli.compare --evaluation evaluations/phase6_smoke
python -m mad_driving.cli.render_episode --evaluation evaluations/phase6_smoke --episode-key <key>
```

Formal plans omit `--smoke`, reference verified method-specific training runs, and
list policy seeds `42, 43, 44, 45, 46`. `compare` and `render_episode` are offline: they
must complete without importing or initializing MetaDrive.

The CLI returns exit code 2 for user/config/artifact-contract errors and a nonzero
runtime error for simulator or model failures. It prints the final published path only
after atomic publication succeeds.

## Error Handling

The implementation fails closed for:

- missing, duplicate, malformed, or incompatible method definitions;
- checkpoints not bound to the requested method and training provenance;
- a B0 checkpoint argument;
- an absent selected-checkpoint record for non-smoke test evaluation;
- test seeds reaching training, curriculum, or checkpoint selection;
- method/track/Shield-contract violations;
- seed or scenario-parameter mismatch across compared methods;
- control fail-safe, Agent contract failure outside the existing isolated fallback,
  invalid action, non-finite telemetry, or contradictory scenario outcome;
- truncated episodes that lack an explicit typed outcome;
- partial, replaced, or hash-mismatched artifacts;
- attempts to overwrite an existing evaluation directory.

Environment/model/frame resources are closed on every error path. A failed evaluation
does not publish a manifest and cannot be consumed by downstream stages.

## Testing Strategy

All production behavior follows test-first RED/GREEN cycles.

### Unit tests

- every method profile and invalid profile/track combination;
- no-op Critic and intentional Agent omissions;
- visible-scene-only B0 TTC thresholds;
- strict evaluation plan and episode-key validation;
- step-record serialization and malformed JSONL rejection;
- all episode metrics, event boundaries, `N/A` denominators, and percentile edges;
- all-level checkpoint ranking and deterministic tie-breaks;
- exact matched-seed comparison rejection cases;
- atomic writer publication, collision refusal, cleanup containment, and hash inventory;
- deterministic CSV, plots, GIF overlays, and Markdown report content;
- CLI argument and exit-code behavior.

### Integration tests

- a deterministic fake Gymnasium environment runs the complete staged pipeline;
- PPO and B0 policy adapters produce actions through the same runner;
- comparison/report/plot/GIF regeneration works after the simulator is closed;
- test artifacts cannot be passed to model selection;
- interrupted simulation leaves no consumable evaluation.

### Real MetaDrive smoke

- one episode for every fixed scenario case with the B0 adapter;
- one short PPO checkpoint evaluation through the Proposed profile;
- matched seeds and scenario parameters are identical across at least two methods;
- one PNG and one GIF are decoded and checked for finite, non-empty dimensions;
- environment reset/step/close and resource cleanup complete headlessly.

The final gate runs Ruff, Ruff format check, mypy, the complete pytest suite with branch
coverage at least 90%, a Phase 6 CLI smoke command, and `git diff --check`.

## Documentation and Architecture Map Follow-up

README gains setup with the `evaluation` extra, method-specific training examples,
checkpoint selection, smoke evaluation, offline comparison, and GIF rendering.
`docs/phase6_implementation_log.md` records exact commands, timings, generated
artifacts, known warnings, and the boundary between infrastructure smoke and the
future formal five-seed run.

After Phase 6 code, tests, PR push, and CI are complete, the separate workspace branch
`feat-architecture-map-flow-cards` is updated. Its canonical architecture-map data,
HTML flow cards, implementation-status labels, Phase 6 artifact flow, CLI cards, and
QA images are regenerated or updated according to that branch's own validation tool.
No Phase 6 production code is copied into that branch. The map update receives its own
commit and push after its validator and presentation checks pass.

## Acceptance Criteria

The Phase 6 infrastructure is complete when:

1. all seven method profiles are strict and covered by tests;
2. B0, B1, B2, and Proposed can use matched final test episode plans;
3. decision comparison is all-monitor and system comparison is all-enforce;
4. checkpoint selection uses a fixed all-level validation suite and never test data;
5. every required metric is produced from fixed oracle or behavior data with explicit
   `N/A` handling;
6. JSONL, summary JSON, episode CSV, comparison CSV, PNG, GIF, and Markdown artifacts
   are generated by a smoke run;
7. downstream artifacts can be regenerated without MetaDrive;
8. malformed or unmatched artifacts are rejected rather than partially compared;
9. the 24-dimensional observation schema remains unchanged;
10. all repository verification gates pass;
11. the Phase 6 branch is pushed as a stacked PR over Phase 5;
12. `feat-architecture-map-flow-cards` is subsequently updated, validated, committed,
    and pushed;
13. the full five-policy-seed benchmark remains explicitly unexecuted until the user
    starts the separate formal-experiment gate.
