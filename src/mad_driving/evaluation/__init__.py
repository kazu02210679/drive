"""Strict, simulator-independent Phase 6 evaluation contracts."""

from mad_driving.evaluation.compare import (
    ComparisonRow,
    build_comparison_rows,
    validate_matched_episodes,
    write_comparison_csv,
    write_eval_metrics_csv,
)
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
    Phase6PpoRunBinding,
    Phase6PublicationPlan,
    PpoRunBinding,
    ScenarioCellId,
    require_phase6_publication_plan,
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
    ValidationPhysicalIdentity,
    discover_checkpoint_candidates,
    select_checkpoint,
    validate_ppo_checkpoint_archive,
    write_selection_artifacts,
)
from mad_driving.evaluation.serialization import (
    load_evaluation_plan,
    load_phase6_publication_plan,
    parse_jsonl_bytes_strict,
    read_jsonl_strict,
    write_jsonl_strict,
)
from mad_driving.evaluation.training_metrics import (
    TrainingMetricPoint,
    extract_training_metrics,
    write_training_metrics_csv,
)
from mad_driving.evaluation.workspace import EvaluationWorkspace

__all__ = [
    "EVALUATION_CASES",
    "REWARD_COMPONENT_KEYS",
    "CheckpointCandidate",
    "CheckpointScore",
    "ComparisonRow",
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
    "Phase6PpoRunBinding",
    "Phase6PublicationPlan",
    "PpoPolicyAdapter",
    "PpoRunBinding",
    "ScenarioCellId",
    "TrainingMetricPoint",
    "ValidationPhysicalIdentity",
    "VisibleTtcRulePolicy",
    "build_comparison_rows",
    "build_formal_plan",
    "build_smoke_plan",
    "discover_checkpoint_candidates",
    "extract_training_metrics",
    "load_evaluation_plan",
    "load_phase6_publication_plan",
    "parse_jsonl_bytes_strict",
    "read_jsonl_strict",
    "reduce_episode",
    "require_phase6_publication_plan",
    "run_evaluation_episode",
    "select_checkpoint",
    "validate_matched_episodes",
    "validate_ppo_checkpoint_archive",
    "write_comparison_csv",
    "write_eval_metrics_csv",
    "write_jsonl_strict",
    "write_selection_artifacts",
    "write_training_metrics_csv",
]
