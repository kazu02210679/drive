"""Strict immutable identities and records for Phase 6 evaluation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Literal, NoReturn, Self, cast, get_args
from unicodedata import category

from pydantic import Field, field_validator, model_validator

from mad_driving.config.models import MethodId, StrictTypedFrozenModel
from mad_driving.interfaces import CollisionKind, CriticReview, RiskClaim
from mad_driving.methods import MethodProfileSnapshot

EvaluationTrack = Literal["decision", "system", "ablation"]
EvaluationRole = Literal["validation", "test"]
ScenarioCellId = Literal[
    "level0_nominal",
    "level1_lead_brake",
    "level2_lead_brake",
    "level2_cut_in",
    "level3_occluded_crossing",
]
ShieldMode = Literal["off", "monitor", "enforce"]

RECORD_SCHEMA_VERSION = 1
RESEARCH_CONTRACT_VERSION = 7
TEST_SEED_START = 20_000
TEST_SEED_STOP = 21_000
VALIDATION_SEED_START = 10_000
VALIDATION_SEED_STOP = 11_000

REWARD_COMPONENT_KEYS = (
    "progress_reward",
    "arrival_reward",
    "collision_penalty",
    "near_miss_penalty",
    "offroad_penalty",
    "rule_violation_penalty",
    "jerk_penalty",
    "unnecessary_brake_penalty",
    "standstill_penalty",
    "shield_intervention_penalty",
)

_METHOD_IDS = frozenset(get_args(MethodId))
_TRACKS = frozenset(get_args(EvaluationTrack))
_ROLES = frozenset(get_args(EvaluationRole))
_CASE_IDS = frozenset(get_args(ScenarioCellId))
_SHIELD_MODES = frozenset(get_args(ShieldMode))
_COLLISION_KINDS = frozenset(get_args(CollisionKind))
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_MAX_ERROR_MESSAGE_LENGTH = 256
_PYTHON_EXCEPTION_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_REPLACED_ERROR_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})


@dataclass(frozen=True)
class EvaluationCase:
    """One fixed scenario cell in the Phase 6 comparison matrix."""

    case_id: ScenarioCellId
    difficulty_level: int
    scenario_id: str

    def __post_init__(self) -> None:
        if self.case_id not in _CASE_IDS:
            raise ValueError("case_id is unknown")
        if type(self.difficulty_level) is not int or not 0 <= self.difficulty_level <= 3:
            raise ValueError("difficulty_level must be an integer from 0 through 3")
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario_id must be a non-empty string")


EVALUATION_CASES = (
    EvaluationCase("level0_nominal", 0, "nominal"),
    EvaluationCase("level1_lead_brake", 1, "lead_brake"),
    EvaluationCase("level2_lead_brake", 2, "lead_brake"),
    EvaluationCase("level2_cut_in", 2, "cut_in"),
    EvaluationCase("level3_occluded_crossing", 3, "occluded_crossing"),
)
_CASES_BY_ID = MappingProxyType({case.case_id: case for case in EVALUATION_CASES})


@dataclass(frozen=True)
class EvaluationEpisodeKey:
    """Complete immutable identity of one evaluation episode."""

    method_id: MethodId
    track: EvaluationTrack
    role: EvaluationRole
    policy_seed: int | None
    case_id: ScenarioCellId
    episode_rng_seed: int

    def __post_init__(self) -> None:
        if self.method_id not in _METHOD_IDS:
            raise ValueError("method_id is unknown")
        if self.track not in _TRACKS:
            raise ValueError("track is unknown")
        if self.role not in _ROLES:
            raise ValueError("role must be validation or test")
        if self.case_id not in _CASE_IDS:
            raise ValueError("case_id is unknown")
        if self.method_id == "b0_rule":
            if self.policy_seed is not None:
                raise ValueError("B0 requires policy_seed=None")
        else:
            _require_int("PPO policy_seed", self.policy_seed, minimum=0)
        _require_episode_seed(self.role, self.episode_rng_seed)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "episode_rng_seed": self.episode_rng_seed,
            "method_id": self.method_id,
            "policy_seed": self.policy_seed,
            "role": self.role,
            "track": self.track,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EvaluationEpisodeKey:
        _require_fields(payload, _EPISODE_KEY_FIELDS, "episode_key")
        return cls(
            method_id=cast(MethodId, payload["method_id"]),
            track=cast(EvaluationTrack, payload["track"]),
            role=cast(EvaluationRole, payload["role"]),
            policy_seed=cast(int | None, payload["policy_seed"]),
            case_id=cast(ScenarioCellId, payload["case_id"]),
            episode_rng_seed=cast(int, payload["episode_rng_seed"]),
        )


_EPISODE_KEY_FIELDS = frozenset(
    {"method_id", "track", "role", "policy_seed", "case_id", "episode_rng_seed"}
)


@dataclass(frozen=True)
class EvaluationRunSpec:
    """One filesystem-independent planned evaluation run."""

    track: EvaluationTrack
    method_id: MethodId
    policy_seed: int | None
    checkpoint_path: str | None
    scenario_cell_id: ScenarioCellId
    episode_index: int
    test_seed: int
    shield_mode: ShieldMode
    is_formal: bool

    def __post_init__(self) -> None:
        if self.track not in _TRACKS:
            raise ValueError("track is unknown")
        if self.method_id not in _METHOD_IDS:
            raise ValueError("method_id is unknown")
        if self.scenario_cell_id not in _CASE_IDS:
            raise ValueError("scenario_cell_id is unknown")
        if self.shield_mode not in _SHIELD_MODES:
            raise ValueError("shield_mode is unknown")
        _require_int("episode_index", self.episode_index, minimum=0)
        _require_int("test_seed", self.test_seed, minimum=TEST_SEED_START)
        if self.test_seed >= TEST_SEED_STOP:
            raise ValueError("test_seed must stay in [20000, 21000)")
        if not isinstance(self.is_formal, bool):
            raise ValueError("is_formal must be boolean")
        if self.method_id == "b0_rule":
            if self.policy_seed is not None or self.checkpoint_path is not None:
                raise ValueError("B0 must not have a policy seed or checkpoint")
        else:
            _require_int("PPO policy_seed", self.policy_seed, minimum=0)
            _require_non_empty_string("checkpoint_path", self.checkpoint_path)

    @property
    def episode_key(self) -> EvaluationEpisodeKey:
        return EvaluationEpisodeKey(
            method_id=self.method_id,
            track=self.track,
            role="test",
            policy_seed=self.policy_seed,
            case_id=self.scenario_cell_id,
            episode_rng_seed=self.test_seed,
        )


class PpoRunBinding(StrictTypedFrozenModel):
    """Plan provenance binding one PPO method seed to its training run."""

    method_id: MethodId
    policy_seed: int = Field(ge=0)
    training_run_dir: Path

    @field_validator("training_run_dir", mode="before")
    @classmethod
    def validate_training_run_dir(cls, value: object) -> Path:
        del cls
        return _strict_non_empty_path(value, "training_run_dir")

    @model_validator(mode="after")
    def validate_ppo_binding(self) -> Self:
        if self.method_id == "b0_rule":
            raise ValueError("PpoRunBinding method_id must identify a PPO method")
        if isinstance(self.policy_seed, bool):
            raise ValueError("policy_seed must be a non-negative integer")
        if not str(self.training_run_dir):
            raise ValueError("training_run_dir must not be empty")
        return self


class EvaluationPlanConfig(StrictTypedFrozenModel):
    """Strict YAML boundary for one smoke or formal evaluation plan."""

    plan_kind: Literal["phase6_smoke", "phase6_formal"]
    evaluation_id: str = Field(min_length=1)
    app_config_path: Path
    episodes_per_case: int = Field(gt=0)
    test_seed_start: int = Field(ge=TEST_SEED_START, lt=TEST_SEED_STOP)
    ppo_run_bindings: tuple[PpoRunBinding, ...]
    capture_episode_keys: tuple[str, ...]

    @field_validator("app_config_path", mode="before")
    @classmethod
    def validate_app_config_path(cls, value: object) -> Path:
        del cls
        return _strict_non_empty_path(value, "app_config_path")

    @field_validator("ppo_run_bindings", "capture_episode_keys", mode="before")
    @classmethod
    def normalize_yaml_sequences(cls, value: object) -> object:
        del cls
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_unique_values(self) -> Self:
        binding_keys = tuple(
            (binding.method_id, binding.policy_seed) for binding in self.ppo_run_bindings
        )
        if len(binding_keys) != len(set(binding_keys)):
            raise ValueError("ppo_run_bindings contain a duplicate method/policy seed key")
        if (
            not all(isinstance(key, str) and key for key in self.capture_episode_keys)
            or len(self.capture_episode_keys) != len(set(self.capture_episode_keys))
        ):
            raise ValueError("capture_episode_keys must contain unique non-empty strings")
        if not str(self.app_config_path):
            raise ValueError("app_config_path must not be empty")
        return self


@dataclass(frozen=True)
class EvaluationStepRecord:
    """Strict serializable record for one evaluation decision step."""

    record_schema_version: int
    research_contract_version: int
    episode_key: EvaluationEpisodeKey
    method_profile: MethodProfileSnapshot
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    step_index: int
    simulation_time_s: float
    decision_interval_s: float
    episode_rng_seed: int
    metadrive_scenario_index: int
    scenario_selection_seed: int
    scenario_parameter_seed: int
    case_id: ScenarioCellId
    scenario_id: str
    difficulty_level: int
    requested_action: int
    required_action: int
    executed_action: int
    unsafe_request: bool
    shield_intervened: bool
    shield_reasons: tuple[str, ...]
    target_speed_mps: float
    ego_speed_mps: float
    ego_longitudinal_acceleration_mps2: float
    route_completion: float
    route_progress_m: float
    lane_offset_m: float
    collision_occurred: bool
    collision_kind: CollisionKind | None
    minimum_actual_ttc_s: float | None
    minimum_actual_stopping_margin_m: float | None
    pre_step_hard_rule_constraint: bool
    post_step_rule_violation_event: bool
    scenario_success: bool
    scenario_failure: bool
    arrived: bool
    off_road: bool
    terminated: bool
    truncated: bool
    cumulative_unnecessary_stop_duration_s: float
    reward_total: float
    reward_components: Mapping[str, float]
    claims: tuple[RiskClaim, ...]
    review: CriticReview
    expected_agent_ids: tuple[str, ...]
    failed_agent_ids: tuple[str, ...]
    errors: tuple[str, ...]
    policy_inference_latency_ms: float
    agent_analysis_latency_ms: float
    shield_latency_ms: float
    total_decision_latency_ms: float
    frame_path: str | None

    def __post_init__(self) -> None:
        _validate_versions(self.record_schema_version, self.research_contract_version)
        episode_key = _canonical_episode_key(self.episode_key)
        profile = _canonical_profile(self.method_profile, episode_key.method_id)
        _validate_checkpoint_identity(
            episode_key.method_id, self.checkpoint_path, self.checkpoint_sha256
        )
        _require_int("step_index", self.step_index, minimum=0)
        _require_non_negative("simulation_time_s", self.simulation_time_s)
        _require_positive("decision_interval_s", self.decision_interval_s)
        _validate_step_seed_and_case_identity(self, episode_key)
        _validate_actions(self)
        shield_reasons = _canonical_strings(
            "shield_reasons", self.shield_reasons, unique=True, non_empty=True
        )
        _require_non_negative("target_speed_mps", self.target_speed_mps)
        _require_non_negative("ego_speed_mps", self.ego_speed_mps)
        _require_finite(
            "ego_longitudinal_acceleration_mps2",
            self.ego_longitudinal_acceleration_mps2,
        )
        _require_probability("route_completion", self.route_completion)
        _require_non_negative("route_progress_m", self.route_progress_m)
        _require_finite("lane_offset_m", self.lane_offset_m)
        _validate_oracle_and_outcomes(self)
        _require_non_negative(
            "cumulative_unnecessary_stop_duration_s",
            self.cumulative_unnecessary_stop_duration_s,
        )
        components = _canonical_reward_components(self.reward_components, self.reward_total)
        claims = _canonical_claims(self.claims)
        review = _canonical_review(self.review)
        expected, failed, errors = _validate_analysis_identity(
            profile, claims, review, self.expected_agent_ids, self.failed_agent_ids, self.errors
        )
        _validate_latencies(self)
        _validate_frame_path(self.frame_path)
        object.__setattr__(self, "episode_key", episode_key)
        object.__setattr__(self, "method_profile", profile)
        object.__setattr__(self, "shield_reasons", shield_reasons)
        object.__setattr__(self, "reward_components", components)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "review", review)
        object.__setattr__(self, "expected_agent_ids", expected)
        object.__setattr__(self, "failed_agent_ids", failed)
        object.__setattr__(self, "errors", errors)

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_analysis_latency_ms": self.agent_analysis_latency_ms,
            "arrived": self.arrived,
            "case_id": self.case_id,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "claims": [_claim_to_dict(claim) for claim in self.claims],
            "collision_kind": self.collision_kind,
            "collision_occurred": self.collision_occurred,
            "cumulative_unnecessary_stop_duration_s": self.cumulative_unnecessary_stop_duration_s,
            "decision_interval_s": self.decision_interval_s,
            "difficulty_level": self.difficulty_level,
            "ego_longitudinal_acceleration_mps2": self.ego_longitudinal_acceleration_mps2,
            "ego_speed_mps": self.ego_speed_mps,
            "episode_key": self.episode_key.to_dict(),
            "episode_rng_seed": self.episode_rng_seed,
            "errors": list(self.errors),
            "executed_action": self.executed_action,
            "expected_agent_ids": list(self.expected_agent_ids),
            "failed_agent_ids": list(self.failed_agent_ids),
            "frame_path": self.frame_path,
            "lane_offset_m": self.lane_offset_m,
            "metadrive_scenario_index": self.metadrive_scenario_index,
            "method_profile": _profile_to_dict(self.method_profile),
            "minimum_actual_stopping_margin_m": self.minimum_actual_stopping_margin_m,
            "minimum_actual_ttc_s": self.minimum_actual_ttc_s,
            "off_road": self.off_road,
            "policy_inference_latency_ms": self.policy_inference_latency_ms,
            "post_step_rule_violation_event": self.post_step_rule_violation_event,
            "pre_step_hard_rule_constraint": self.pre_step_hard_rule_constraint,
            "record_schema_version": self.record_schema_version,
            "requested_action": self.requested_action,
            "required_action": self.required_action,
            "research_contract_version": self.research_contract_version,
            "review": _review_to_dict(self.review),
            "reward_components": dict(self.reward_components),
            "reward_total": self.reward_total,
            "route_completion": self.route_completion,
            "route_progress_m": self.route_progress_m,
            "scenario_failure": self.scenario_failure,
            "scenario_id": self.scenario_id,
            "scenario_parameter_seed": self.scenario_parameter_seed,
            "scenario_selection_seed": self.scenario_selection_seed,
            "scenario_success": self.scenario_success,
            "shield_intervened": self.shield_intervened,
            "shield_latency_ms": self.shield_latency_ms,
            "shield_reasons": list(self.shield_reasons),
            "simulation_time_s": self.simulation_time_s,
            "step_index": self.step_index,
            "target_speed_mps": self.target_speed_mps,
            "terminated": self.terminated,
            "total_decision_latency_ms": self.total_decision_latency_ms,
            "truncated": self.truncated,
            "unsafe_request": self.unsafe_request,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EvaluationStepRecord:
        _require_fields(payload, _STEP_FIELDS, "EvaluationStepRecord")
        values = dict(payload)
        values["episode_key"] = EvaluationEpisodeKey.from_dict(
            _require_mapping(values["episode_key"], "episode_key")
        )
        values["method_profile"] = _profile_from_dict(
            _require_mapping(values["method_profile"], "method_profile")
        )
        values["claims"] = tuple(
            _claim_from_dict(_require_mapping(item, "claim"))
            for item in _require_sequence(values["claims"], "claims")
        )
        values["review"] = _review_from_dict(
            _require_mapping(values["review"], "review")
        )
        values["reward_components"] = _require_mapping(
            values["reward_components"], "reward_components"
        )
        for name in ("shield_reasons", "expected_agent_ids", "failed_agent_ids", "errors"):
            values[name] = tuple(_require_sequence(values[name], name))
        try:
            return cls(**values)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("EvaluationStepRecord fields are malformed") from error


@dataclass(frozen=True)
class EvaluationEpisodeRecord:
    """Strict completed-episode summary, distinct from later metric rows."""

    record_schema_version: int
    research_contract_version: int
    episode_key: EvaluationEpisodeKey
    method_profile: MethodProfileSnapshot
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    episode_rng_seed: int
    metadrive_scenario_index: int
    scenario_selection_seed: int
    scenario_parameter_seed: int
    case_id: ScenarioCellId
    scenario_id: str
    difficulty_level: int
    sampled_scenario_parameters: Mapping[str, object]
    step_count: int
    final_step_index: int
    simulated_duration_s: float
    cumulative_reward: float
    collision_occurred: bool
    collision_kind: CollisionKind | None
    scenario_success: bool
    scenario_failure: bool
    arrived: bool
    off_road: bool
    terminated: bool
    truncated: bool
    complete: bool

    def __post_init__(self) -> None:
        _validate_versions(self.record_schema_version, self.research_contract_version)
        episode_key = _canonical_episode_key(self.episode_key)
        profile = _canonical_profile(self.method_profile, episode_key.method_id)
        _validate_checkpoint_identity(
            episode_key.method_id, self.checkpoint_path, self.checkpoint_sha256
        )
        _validate_episode_seed_and_case_identity(self, episode_key)
        parameters = _freeze_json_object(
            self.sampled_scenario_parameters, "sampled_scenario_parameters"
        )
        _require_int("step_count", self.step_count, minimum=1)
        _require_int("final_step_index", self.final_step_index, minimum=0)
        if self.final_step_index != self.step_count - 1:
            raise ValueError("final_step_index must equal step_count - 1")
        _require_non_negative("simulated_duration_s", self.simulated_duration_s)
        _require_finite("cumulative_reward", self.cumulative_reward)
        _validate_collision(self.collision_occurred, self.collision_kind)
        _validate_boolean_fields(
            self,
            (
                "scenario_success",
                "scenario_failure",
                "arrived",
                "off_road",
                "terminated",
                "truncated",
                "complete",
            ),
        )
        if self.scenario_success and self.scenario_failure:
            raise ValueError("scenario success and failure cannot both be true")
        if not self.complete:
            raise ValueError("complete must be True")
        if not (self.terminated or self.truncated):
            raise ValueError("complete episode must have a terminal flag")
        object.__setattr__(self, "episode_key", episode_key)
        object.__setattr__(self, "method_profile", profile)
        object.__setattr__(self, "sampled_scenario_parameters", parameters)

    def to_dict(self) -> dict[str, object]:
        return {
            "arrived": self.arrived,
            "case_id": self.case_id,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "collision_kind": self.collision_kind,
            "collision_occurred": self.collision_occurred,
            "complete": self.complete,
            "cumulative_reward": self.cumulative_reward,
            "difficulty_level": self.difficulty_level,
            "episode_key": self.episode_key.to_dict(),
            "episode_rng_seed": self.episode_rng_seed,
            "final_step_index": self.final_step_index,
            "metadrive_scenario_index": self.metadrive_scenario_index,
            "method_profile": _profile_to_dict(self.method_profile),
            "off_road": self.off_road,
            "record_schema_version": self.record_schema_version,
            "research_contract_version": self.research_contract_version,
            "sampled_scenario_parameters": _thaw_json(self.sampled_scenario_parameters),
            "scenario_failure": self.scenario_failure,
            "scenario_id": self.scenario_id,
            "scenario_parameter_seed": self.scenario_parameter_seed,
            "scenario_selection_seed": self.scenario_selection_seed,
            "scenario_success": self.scenario_success,
            "simulated_duration_s": self.simulated_duration_s,
            "step_count": self.step_count,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EvaluationEpisodeRecord:
        _require_fields(payload, _EPISODE_FIELDS, "EvaluationEpisodeRecord")
        values = dict(payload)
        values["episode_key"] = EvaluationEpisodeKey.from_dict(
            _require_mapping(values["episode_key"], "episode_key")
        )
        values["method_profile"] = _profile_from_dict(
            _require_mapping(values["method_profile"], "method_profile")
        )
        values["sampled_scenario_parameters"] = _require_mapping(
            values["sampled_scenario_parameters"], "sampled_scenario_parameters"
        )
        try:
            return cls(**values)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("EvaluationEpisodeRecord fields are malformed") from error


_STEP_FIELDS = frozenset(EvaluationStepRecord.__dataclass_fields__)
_EPISODE_FIELDS = frozenset(EvaluationEpisodeRecord.__dataclass_fields__)
_PROFILE_FIELDS = frozenset(
    {"method_id", "policy_kind", "specialist_ids", "critic_enabled", "shield_mode"}
)
_CLAIM_FIELDS = frozenset(RiskClaim.__dataclass_fields__)
_REVIEW_FIELDS = frozenset(CriticReview.__dataclass_fields__)


class _FrozenFloatMapping(dict[str, float]):
    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("mapping is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, value: object, /) -> Self:  # type: ignore[override,misc]
        self._immutable(value)


class _FrozenJsonMapping(dict[str, object]):
    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("mapping is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, value: object, /) -> Self:  # type: ignore[override,misc]
        self._immutable(value)


def _require_fields(payload: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        missing = sorted(expected.difference(payload)) if isinstance(payload, Mapping) else []
        extra = sorted(set(payload).difference(expected)) if isinstance(payload, Mapping) else []
        raise ValueError(f"{name} fields are invalid; missing={missing!r}, extra={extra!r}")


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return cast(Sequence[object], value)


def _require_int(
    name: str, value: object, *, minimum: int, maximum: int | None = None
) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        suffix = f" through {maximum}" if maximum is not None else f" >= {minimum}"
        raise ValueError(f"{name} must be an integer{suffix}")
    return value


def _require_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _require_non_negative(name: str, value: object) -> float:
    normalized = _require_finite(name, value)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _require_positive(name: str, value: object) -> float:
    normalized = _require_finite(name, value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _require_probability(name: str, value: object) -> float:
    normalized = _require_finite(name, value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return normalized


def _require_non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_non_empty_path(value: object, name: str) -> Path:
    if isinstance(value, Path):
        raw_value = str(value)
    elif isinstance(value, str):
        raw_value = value
    else:
        raise ValueError(f"{name} must be a non-empty string path")
    if not raw_value.strip() or raw_value == ".":
        raise ValueError(f"{name} must be a non-empty string path")
    return Path(raw_value)


def _require_episode_seed(role: EvaluationRole, seed: object) -> int:
    seed_value = _require_int("episode_rng_seed", seed, minimum=0)
    bounds = (
        (VALIDATION_SEED_START, VALIDATION_SEED_STOP)
        if role == "validation"
        else (TEST_SEED_START, TEST_SEED_STOP)
    )
    if not bounds[0] <= seed_value < bounds[1]:
        raise ValueError(f"episode_rng_seed must stay in [{bounds[0]}, {bounds[1]})")
    return seed_value


def _canonical_strings(
    name: str, values: object, *, unique: bool, non_empty: bool
) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        result: tuple[object, ...] = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of strings") from error
    if not all(isinstance(value, str) for value in result):
        raise ValueError(f"{name} must contain only strings")
    normalized = cast(tuple[str, ...], result)
    if non_empty and not all(normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    if unique and len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _validate_versions(schema_version: object, contract_version: object) -> None:
    if type(schema_version) is not int or schema_version != RECORD_SCHEMA_VERSION:
        raise ValueError("record_schema_version must equal 1")
    if type(contract_version) is not int or contract_version != RESEARCH_CONTRACT_VERSION:
        raise ValueError("research_contract_version must equal 7")


def _canonical_episode_key(value: object) -> EvaluationEpisodeKey:
    if not isinstance(value, EvaluationEpisodeKey):
        raise ValueError("episode_key must be an EvaluationEpisodeKey")
    return EvaluationEpisodeKey.from_dict(value.to_dict())


def _canonical_profile(value: object, method_id: MethodId) -> MethodProfileSnapshot:
    if not isinstance(value, MethodProfileSnapshot):
        raise ValueError("method_profile must be a MethodProfileSnapshot")
    expected = MethodProfileSnapshot.from_method_id(method_id)
    if value != expected:
        raise ValueError("method_profile must match episode_key.method_id")
    return expected


def _profile_to_dict(profile: MethodProfileSnapshot) -> dict[str, object]:
    return {
        "critic_enabled": profile.critic_enabled,
        "method_id": profile.method_id,
        "policy_kind": profile.policy_kind,
        "shield_mode": profile.shield_mode,
        "specialist_ids": list(profile.specialist_ids),
    }


def _profile_from_dict(payload: Mapping[str, object]) -> MethodProfileSnapshot:
    _require_fields(payload, _PROFILE_FIELDS, "method_profile")
    try:
        return MethodProfileSnapshot(
            method_id=cast(str, payload["method_id"]),
            policy_kind=cast(str, payload["policy_kind"]),
            specialist_ids=tuple(
                cast(str, item)
                for item in _require_sequence(payload["specialist_ids"], "specialist_ids")
            ),
            critic_enabled=cast(bool, payload["critic_enabled"]),
            shield_mode=cast(str, payload["shield_mode"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("method_profile is malformed") from error


def _validate_checkpoint_identity(
    method_id: MethodId, checkpoint_path: object, checkpoint_sha256: object
) -> None:
    if method_id == "b0_rule":
        if checkpoint_path is not None or checkpoint_sha256 is not None:
            raise ValueError("B0 checkpoint path and SHA-256 must both be absent")
        return
    _require_non_empty_string("checkpoint_path", checkpoint_path)
    if not isinstance(checkpoint_sha256, str) or not _SHA256_PATTERN.fullmatch(checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")


def _validate_case_identity(
    case_id: object, scenario_id: object, difficulty_level: object
) -> None:
    if case_id not in _CASES_BY_ID:
        raise ValueError("case_id is unknown")
    _require_int("difficulty_level", difficulty_level, minimum=0, maximum=3)
    case = _CASES_BY_ID[cast(ScenarioCellId, case_id)]
    if scenario_id != case.scenario_id or difficulty_level != case.difficulty_level:
        raise ValueError("case_id, scenario_id, and difficulty_level are inconsistent")


def _validate_step_seed_and_case_identity(
    record: EvaluationStepRecord, episode_key: EvaluationEpisodeKey
) -> None:
    if record.episode_rng_seed != episode_key.episode_rng_seed:
        raise ValueError("episode_rng_seed must match episode_key")
    if record.case_id != episode_key.case_id:
        raise ValueError("case_id must match episode_key")
    for name in (
        "metadrive_scenario_index",
        "scenario_selection_seed",
        "scenario_parameter_seed",
    ):
        _require_int(name, getattr(record, name), minimum=0)
    _validate_case_identity(record.case_id, record.scenario_id, record.difficulty_level)


def _validate_episode_seed_and_case_identity(
    record: EvaluationEpisodeRecord, episode_key: EvaluationEpisodeKey
) -> None:
    if record.episode_rng_seed != episode_key.episode_rng_seed:
        raise ValueError("episode_rng_seed must match episode_key")
    if record.case_id != episode_key.case_id:
        raise ValueError("case_id must match episode_key")
    for name in (
        "metadrive_scenario_index",
        "scenario_selection_seed",
        "scenario_parameter_seed",
    ):
        _require_int(name, getattr(record, name), minimum=0)
    _validate_case_identity(record.case_id, record.scenario_id, record.difficulty_level)


def _validate_actions(record: EvaluationStepRecord) -> None:
    for name in ("requested_action", "required_action", "executed_action"):
        _require_int(name, getattr(record, name), minimum=0, maximum=3)
    _validate_boolean_fields(record, ("unsafe_request", "shield_intervened"))
    if record.unsafe_request != (record.required_action > record.requested_action):
        raise ValueError("unsafe_request is inconsistent with requested and required actions")
    if record.shield_intervened != (record.executed_action != record.requested_action):
        raise ValueError("shield_intervened is inconsistent with requested and executed actions")


def _validate_collision(occurred: object, kind: object) -> None:
    if not isinstance(occurred, bool):
        raise ValueError("collision_occurred must be boolean")
    if kind is not None and kind not in _COLLISION_KINDS:
        raise ValueError("collision_kind is unknown")
    if occurred != (kind is not None):
        raise ValueError("collision_occurred and collision_kind are inconsistent")


def _validate_oracle_and_outcomes(record: EvaluationStepRecord) -> None:
    _validate_collision(record.collision_occurred, record.collision_kind)
    if record.minimum_actual_ttc_s is not None:
        _require_non_negative("minimum_actual_ttc_s", record.minimum_actual_ttc_s)
    if record.minimum_actual_stopping_margin_m is not None:
        _require_finite(
            "minimum_actual_stopping_margin_m", record.minimum_actual_stopping_margin_m
        )
    _validate_boolean_fields(
        record,
        (
            "pre_step_hard_rule_constraint",
            "post_step_rule_violation_event",
            "scenario_success",
            "scenario_failure",
            "arrived",
            "off_road",
            "terminated",
            "truncated",
        ),
    )
    if record.scenario_success and record.scenario_failure:
        raise ValueError("scenario success and failure cannot both be true")


def _validate_boolean_fields(value: object, names: tuple[str, ...]) -> None:
    for name in names:
        if not isinstance(getattr(value, name), bool):
            raise ValueError(f"{name} must be boolean")


def _canonical_reward_components(
    values: object, reward_total: object
) -> Mapping[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(REWARD_COMPONENT_KEYS):
        raise ValueError("reward_components must contain exactly the ten required keys")
    normalized = {
        name: _require_finite(f"reward_components.{name}", values[name])
        for name in REWARD_COMPONENT_KEYS
    }
    total = _require_finite("reward_total", reward_total)
    if math.fsum(normalized.values()) != total:
        raise ValueError("reward_total must equal the sum of reward_components")
    return _FrozenFloatMapping(normalized)


def _canonical_claims(values: object) -> tuple[RiskClaim, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise ValueError("claims must contain RiskClaim values")
    try:
        claims = tuple(_canonical_claim(value) for value in values)
    except TypeError as error:
        raise ValueError("claims must contain RiskClaim values") from error
    claim_ids = tuple(claim.claim_id for claim in claims)
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("claims must have unique claim_id values")
    return claims


def _canonical_claim(value: object) -> RiskClaim:
    if not isinstance(value, RiskClaim):
        raise ValueError("claims must contain RiskClaim values")
    return _claim_from_dict(_claim_to_dict(value))


def _claim_to_dict(claim: RiskClaim) -> dict[str, Any]:
    return {
        "agent_id": claim.agent_id,
        "assumptions": list(claim.assumptions),
        "claim_id": claim.claim_id,
        "confidence": claim.confidence,
        "event_type": claim.event_type,
        "evidence": list(claim.evidence),
        "hard_stop_required": claim.hard_stop_required,
        "min_ttc_s": claim.min_ttc_s,
        "probability": claim.probability,
        "recommended_max_speed_mps": claim.recommended_max_speed_mps,
        "severity": claim.severity,
        "stopping_margin_m": claim.stopping_margin_m,
        "target_actor_id": claim.target_actor_id,
        "time_horizon_s": claim.time_horizon_s,
        "valid_until_step": claim.valid_until_step,
    }


def _claim_from_dict(payload: Mapping[str, object]) -> RiskClaim:
    _require_fields(payload, _CLAIM_FIELDS, "RiskClaim")
    values = dict(payload)
    try:
        for name in ("claim_id", "agent_id", "event_type"):
            _require_non_empty_string(name, values[name])
        target_actor_id = values["target_actor_id"]
        if target_actor_id is not None:
            _require_non_empty_string("target_actor_id", target_actor_id)
        for name in ("confidence", "severity"):
            _require_probability(name, values[name])
        probability = values["probability"]
        if probability is not None:
            _require_probability("probability", probability)
        _require_non_negative("time_horizon_s", values["time_horizon_s"])
        minimum_ttc = values["min_ttc_s"]
        if minimum_ttc is not None:
            _require_non_negative("min_ttc_s", minimum_ttc)
        stopping_margin = values["stopping_margin_m"]
        if stopping_margin is not None:
            _require_finite("stopping_margin_m", stopping_margin)
        _require_non_negative(
            "recommended_max_speed_mps", values["recommended_max_speed_mps"]
        )
        if not isinstance(values["hard_stop_required"], bool):
            raise ValueError("hard_stop_required must be boolean")
        _require_int("valid_until_step", values["valid_until_step"], minimum=0)
        values["evidence"] = tuple(_require_sequence(values["evidence"], "evidence"))
        values["assumptions"] = tuple(
            _require_sequence(values["assumptions"], "assumptions")
        )
        return RiskClaim(**values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("RiskClaim is malformed") from error


def _canonical_review(value: object) -> CriticReview:
    if not isinstance(value, CriticReview):
        raise ValueError("review must be a CriticReview")
    return _review_from_dict(_review_to_dict(value))


def _review_to_dict(review: CriticReview) -> dict[str, Any]:
    return {
        "challenged_claim_ids": list(review.challenged_claim_ids),
        "conflict_score": review.conflict_score,
        "max_severity": review.max_severity,
        "reasons": list(review.reasons),
        "supported_agent_ids": list(review.supported_agent_ids),
        "unresolved_conflict": review.unresolved_conflict,
    }


def _review_from_dict(payload: Mapping[str, object]) -> CriticReview:
    _require_fields(payload, _REVIEW_FIELDS, "CriticReview")
    values = dict(payload)
    try:
        _require_probability("conflict_score", values["conflict_score"])
        _require_probability("max_severity", values["max_severity"])
        if not isinstance(values["unresolved_conflict"], bool):
            raise ValueError("unresolved_conflict must be boolean")
        for name in ("supported_agent_ids", "challenged_claim_ids", "reasons"):
            values[name] = tuple(_require_sequence(values[name], name))
        return CriticReview(**values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("CriticReview is malformed") from error


def _validate_analysis_identity(
    profile: MethodProfileSnapshot,
    claims: tuple[RiskClaim, ...],
    review: CriticReview,
    expected_values: object,
    failed_values: object,
    error_values: object,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    expected = _canonical_strings(
        "expected_agent_ids", expected_values, unique=True, non_empty=True
    )
    if expected != profile.specialist_ids:
        raise ValueError("expected_agent_ids must equal the method profile specialists")
    failed = _canonical_strings("failed_agent_ids", failed_values, unique=True, non_empty=True)
    errors = _canonical_strings("errors", error_values, unique=True, non_empty=False)
    if not set(failed).issubset(expected):
        raise ValueError("failed_agent_ids must be a subset of expected_agent_ids")
    if not {claim.agent_id for claim in claims}.issubset(expected):
        raise ValueError("claim agent_ids must be a subset of expected_agent_ids")
    if not set(review.supported_agent_ids).issubset(expected):
        raise ValueError("review supported_agent_ids must be a subset of expected_agent_ids")
    if not set(review.challenged_claim_ids).issubset({claim.claim_id for claim in claims}):
        raise ValueError("review challenged_claim_ids must identify recorded claims")
    _validate_error_mapping(failed, errors)
    return expected, failed, errors


def _validate_error_mapping(failed: tuple[str, ...], errors: tuple[str, ...]) -> None:
    if len(failed) != len(errors):
        raise ValueError("errors must map one-to-one to failed_agent_ids")
    for agent_id, error in zip(failed, errors, strict=True):
        if not error.startswith(f"{agent_id}:"):
            raise ValueError("errors must match failed_agent_ids in order")
        exception_type, separator, message = error[len(agent_id) + 1 :].partition(":")
        if not exception_type or not separator:
            raise ValueError("errors must use agent:type:message format")
        if not _PYTHON_EXCEPTION_NAME_PATTERN.fullmatch(exception_type):
            raise ValueError("errors exception type must be a bounded sanitized Python name")
        if len(message) > _MAX_ERROR_MESSAGE_LENGTH or any(
            category(character) in _REPLACED_ERROR_CATEGORIES for character in message
        ):
            raise ValueError("errors must contain bounded sanitized messages")


def _validate_latencies(record: EvaluationStepRecord) -> None:
    components = tuple(
        _require_non_negative(name, getattr(record, name))
        for name in (
            "policy_inference_latency_ms",
            "agent_analysis_latency_ms",
            "shield_latency_ms",
        )
    )
    total = _require_non_negative("total_decision_latency_ms", record.total_decision_latency_ms)
    if any(total < component for component in components):
        raise ValueError("total_decision_latency_ms must be at least every latency component")


def _validate_frame_path(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError("frame_path must be a non-empty relative path or None")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise ValueError("frame_path must be relative and must not contain '..' or a drive")


def _freeze_json_object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return _FrozenJsonMapping({key: _freeze_json(item, name) for key, item in value.items()})


def _freeze_json(value: object, name: str) -> object:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite JSON values")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_object(value, name)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item, name) for item in value)
    raise ValueError(f"{name} must contain only JSON values")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "EVALUATION_CASES",
    "RECORD_SCHEMA_VERSION",
    "RESEARCH_CONTRACT_VERSION",
    "REWARD_COMPONENT_KEYS",
    "EvaluationCase",
    "EvaluationEpisodeKey",
    "EvaluationEpisodeRecord",
    "EvaluationPlanConfig",
    "EvaluationRole",
    "EvaluationRunSpec",
    "EvaluationStepRecord",
    "EvaluationTrack",
    "PpoRunBinding",
    "ScenarioCellId",
    "ShieldMode",
]
