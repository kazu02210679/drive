"""Immutable result of a Safety Shield decision."""

from dataclasses import dataclass

from mad_driving.control.actions import DrivingAction


@dataclass(frozen=True)
class ShieldResult:
    """Record the requested, required, and actually executed actions."""

    requested_action: DrivingAction
    required_action: DrivingAction
    executed_action: DrivingAction
    intervention_required: bool
    intervened: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("requested_action", "required_action", "executed_action"):
            object.__setattr__(self, name, DrivingAction(getattr(self, name)))
        if self.intervention_required != (self.required_action > self.requested_action):
            raise ValueError("intervention_required is inconsistent")
        if self.intervened != (self.executed_action != self.requested_action):
            raise ValueError("intervened is inconsistent")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("reasons must be duplicate-free")
