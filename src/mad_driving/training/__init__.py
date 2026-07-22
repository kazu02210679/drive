"""Lazy public training API that does not import PPO until it is requested."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mad_driving.training.callbacks import CurriculumEvalCallback, RewardComponentsCallback
    from mad_driving.training.curriculum import (
        CurriculumController,
        CurriculumState,
        read_curriculum_state,
        write_curriculum_state,
    )
    from mad_driving.training.metadata import (
        MethodProfileSnapshot,
        ResumeMetadata,
        RunMetadata,
        sha256_file,
        validate_resume_contract,
    )
    from mad_driving.training.train import TrainingResult, run_training

_EXPORT_MODULES = {
    "CurriculumController": "mad_driving.training.curriculum",
    "CurriculumEvalCallback": "mad_driving.training.callbacks",
    "CurriculumState": "mad_driving.training.curriculum",
    "MethodProfileSnapshot": "mad_driving.training.metadata",
    "ResumeMetadata": "mad_driving.training.metadata",
    "RewardComponentsCallback": "mad_driving.training.callbacks",
    "RunMetadata": "mad_driving.training.metadata",
    "TrainingResult": "mad_driving.training.train",
    "read_curriculum_state": "mad_driving.training.curriculum",
    "run_training": "mad_driving.training.train",
    "sha256_file": "mad_driving.training.metadata",
    "validate_resume_contract": "mad_driving.training.metadata",
    "write_curriculum_state": "mad_driving.training.curriculum",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORT_MODULES))


__all__ = [
    "CurriculumController",
    "CurriculumEvalCallback",
    "CurriculumState",
    "MethodProfileSnapshot",
    "ResumeMetadata",
    "RewardComponentsCallback",
    "RunMetadata",
    "TrainingResult",
    "read_curriculum_state",
    "run_training",
    "sha256_file",
    "validate_resume_contract",
    "write_curriculum_state",
]
