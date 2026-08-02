"""Deterministic nominal constant-acceleration motion analysis."""

from dataclasses import dataclass
from math import exp

from mad_driving.agents.claim_factory import claim_id, neutral_claim
from mad_driving.agents.kinematics import (
    project_vector,
    rectangular_clearance,
    relative_position,
    sample_times,
)
from mad_driving.config.models import NominalAgentConfig
from mad_driving.interfaces import ActorState, RiskClaim, SceneSnapshot


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(max(value, lower), upper)


@dataclass(frozen=True)
class _Candidate:
    claim: RiskClaim
    minimum_clearance_m: float


class NominalMotionAgent:
    """Predict likely observed-actor motion over a finite horizon."""

    agent_id = "nominal"

    def __init__(self, config: NominalAgentConfig) -> None:
        self._config = config

    def analyze(self, snapshot: SceneSnapshot) -> RiskClaim:
        candidates = tuple(
            candidate
            for actor in sorted(snapshot.actors, key=lambda item: item.actor_id)
            if (candidate := self._evaluate_actor(snapshot, actor)) is not None
        )
        if not candidates:
            return neutral_claim(self.agent_id, snapshot)
        return min(
            candidates,
            key=lambda candidate: (
                -candidate.claim.severity,
                -(candidate.claim.probability or 0.0),
                candidate.minimum_clearance_m,
                candidate.claim.target_actor_id or "",
            ),
        ).claim

    def _evaluate_actor(self, snapshot: SceneSnapshot, actor: ActorState) -> _Candidate | None:
        if actor.same_lane and actor.relative_longitudinal_m <= 0.0:
            return None

        velocity = project_vector(actor.velocity_xy_mps, snapshot.ego.heading_rad)
        acceleration = project_vector(actor.acceleration_xy_mps2, snapshot.ego.heading_rad)
        relative_velocity = (velocity[0] - snapshot.ego.speed_mps, velocity[1])
        relative_acceleration = (
            acceleration[0] - snapshot.ego.acceleration_mps2,
            acceleration[1],
        )
        positions = tuple(
            (
                time_s,
                relative_position(
                    (actor.relative_longitudinal_m, actor.relative_lateral_m),
                    relative_velocity,
                    relative_acceleration,
                    time_s,
                ),
            )
            for time_s in sample_times(self._config.horizon_s, self._config.time_step_s)
        )

        is_crossing = actor.actor_type == "crossing_actor"
        is_cut_in = not actor.same_lane and any(
            abs(position[1]) <= self._config.lane_half_width_m for _, position in positions
        )
        if not actor.same_lane and not is_crossing and not is_cut_in:
            return None

        longitudinal_envelope = (
            0.5 * (self._config.ego_length_m + actor.length_m) + self._config.longitudinal_buffer_m
        )
        lateral_envelope = (
            0.5 * (self._config.ego_width_m + actor.width_m) + self._config.lateral_buffer_m
        )
        clearances = tuple(
            (
                time_s,
                rectangular_clearance(position, longitudinal_envelope, lateral_envelope),
            )
            for time_s, position in positions
        )
        minimum_clearance = min(clearance for _, clearance in clearances)
        ttc = next(
            (time_s for time_s, clearance in clearances if clearance == 0.0),
            None,
        )
        closing_speed = max(snapshot.ego.speed_mps - velocity[0], 0.0)
        ttc_term = exp(-ttc / self._config.probability_ttc_scale_s) if ttc is not None else 0.0
        distance_term = exp(-minimum_clearance / self._config.probability_distance_scale_m)
        closing_term = _clip(closing_speed / 10.0)
        probability = _clip(0.65 * ttc_term + 0.25 * distance_term + 0.10 * closing_term)
        recommended_speed = snapshot.ego.speed_limit_mps * (1.0 - 0.75 * probability)
        event_type = self._event_type(actor, is_crossing)
        ttc_evidence = "none" if ttc is None else f"{ttc:.6f}"
        claim = RiskClaim(
            claim_id=claim_id(self.agent_id, snapshot, event_type, actor.actor_id),
            agent_id=self.agent_id,
            event_type=event_type,
            target_actor_id=actor.actor_id,
            probability=probability,
            confidence=0.9,
            severity=probability,
            time_horizon_s=self._config.horizon_s,
            min_ttc_s=ttc,
            stopping_margin_m=None,
            recommended_max_speed_mps=recommended_speed,
            hard_stop_required=False,
            evidence=(
                f"minimum_clearance_m={minimum_clearance:.6f}",
                f"ttc_s={ttc_evidence}",
            ),
            assumptions=("constant_acceleration",),
            valid_until_step=snapshot.step_index,
        )
        return _Candidate(claim=claim, minimum_clearance_m=minimum_clearance)

    @staticmethod
    def _event_type(actor: ActorState, is_crossing: bool) -> str:
        if is_crossing:
            return "nominal_crossing"
        if actor.same_lane:
            return "nominal_lead"
        return "nominal_cut_in"
