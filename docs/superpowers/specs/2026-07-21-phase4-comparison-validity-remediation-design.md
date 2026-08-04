# Phase 4 Comparison-Validity Remediation

## Status

This design supersedes the reward, Agent-missing, low-level fail-safe, and research-contract
parts of the Phase 4 research-validity hardening design. The 24-dimensional Coordinator
Observation and observation schema version 1 remain unchanged.

## Comparison-independent reward

`RewardContext` contains only the previous and next `SceneFrame`, executed action, Shield
intervention status, and decision interval. It does not accept specialist claims, reviews,
or failure status. `PrivilegedWorldState.minimum_actual_ttc_s` is a fixed oracle computed
from every simulator-truth actor, including occluded actors, using constant current velocity
and first entry into the ego-aligned rectangular collision envelope. The pre-step privileged
`hard_rule_constraint` drives rule and unnecessary-brake validity. The post-step oracle TTC
drives near-miss penalty. Therefore identical physical transitions have identical rewards
for B1, B2, Proposed, and all Agent ablations.

Vehicle, object, sidewalk, and building collisions share the vehicle collision penalty.
Crossing-actor collisions retain their dedicated larger penalty.

## Ablation-aware Shield

`AgentAnalysisResult` carries both `expected_agent_ids` and `failed_agent_ids`.
`AgentSuite` derives expected IDs from configured non-`None` specialists. The Shield does
not count specialists absent from the expected set as missing. An expected specialist with
no valid Claim, or one explicitly listed as failed, activates the configured one/multiple
missing safety floor.

## Internal failures

An active low-level controller fail-safe is not returned as a normal Gymnasium transition.
`MultiAgentSpeedEnv` closes its simulator and raises `RuntimeError`, preventing PPO from
crediting the requested high-level action for emergency-braking behavior. Scenario outcomes
require strict booleans, cannot be success and failure simultaneously, and transitions
validate their owned state and outcome types at construction.

## Versioning

These changes alter the learning objective and transition contract. New artifacts use
`research_contract_version=4`. Observation shape, dtype, feature order, and
`observation_schema_version=1` are unchanged. Contracts 1 through 3 cannot be resumed into
contract-4 training runs.
