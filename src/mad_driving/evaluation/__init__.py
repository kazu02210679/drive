"""Strict, simulator-independent Phase 6 evaluation contracts."""

from mad_driving.evaluation.metrics import (
    EpisodeMetricRecord,
    EpisodeMetrics,
    reduce_episode,
)
from mad_driving.evaluation.models import (
    EVALUATION_CASES,
    REWARD_COMPONENT_KEYS,
    EvaluationCase,
    EvaluationEpisodeKey,
    EvaluationEpisodeRecord,
    EvaluationPlanConfig,
    EvaluationRunSpec,
    EvaluationStepRecord,
    EvaluationTrack,
    PpoRunBinding,
    ScenarioCellId,
)
from mad_driving.evaluation.plans import build_formal_plan, build_smoke_plan
from mad_driving.evaluation.policies import (
    EvaluationPolicy,
    PpoPolicyAdapter,
    VisibleTtcRulePolicy,
)
from mad_driving.evaluation.runner import EvaluationRunResult, run_evaluation_episode
from mad_driving.evaluation.selection import (
    CheckpointCandidate,
    CheckpointScore,
    discover_checkpoint_candidates,
    select_checkpoint,
    write_selection_artifacts,
)
from mad_driving.evaluation.serialization import (
    load_evaluation_plan,
    read_jsonl_strict,
    write_jsonl_strict,
)
from mad_driving.evaluation.workspace import EvaluationWorkspace

__all__ = [
    "EVALUATION_CASES",
    "REWARD_COMPONENT_KEYS",
    "CheckpointCandidate",
    "CheckpointScore",
    "EpisodeMetricRecord",
    "EpisodeMetrics",
    "EvaluationCase",
    "EvaluationEpisodeKey",
    "EvaluationEpisodeRecord",
    "EvaluationPlanConfig",
    "EvaluationPolicy",
    "EvaluationRunResult",
    "EvaluationRunSpec",
    "EvaluationStepRecord",
    "EvaluationTrack",
    "EvaluationWorkspace",
    "PpoPolicyAdapter",
    "PpoRunBinding",
    "ScenarioCellId",
    "VisibleTtcRulePolicy",
    "build_formal_plan",
    "build_smoke_plan",
    "discover_checkpoint_candidates",
    "load_evaluation_plan",
    "read_jsonl_strict",
    "reduce_episode",
    "run_evaluation_episode",
    "select_checkpoint",
    "write_jsonl_strict",
    "write_selection_artifacts",
]
