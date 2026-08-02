"""Deterministic worst-case braking, crossing, and occlusion analysis."""

from dataclasses import dataclass
from math import exp, sqrt

from mad_driving.agents.claim_factory import claim_id, neutral_claim, ordered_claims
from mad_driving.agents.kinematics import (
    project_vector,
    safe_speed_for_distance,
    stopping_distance,
)
from mad_driving.config.models import HazardAgentConfig
from mad_driving.interfaces import ActorState, RiskClaim, SceneObservation


@dataclass(frozen=True)
class _Candidate:
    claim: RiskClaim


def _lead_braking_ttc(
    *,
    bumper_gap_m: float,
    ego_speed_mps: float,
    lead_speed_mps: float,
    lead_deceleration_mps2: float,
) -> float | None:
    """Return TTC for a constant-speed ego and a maximally braking lead."""

    if bumper_gap_m <= 0.0:
        return 0.0
    if ego_speed_mps <= 0.0:
        return None

    braking = abs(lead_deceleration_mps2)
    lead_stop_time = lead_speed_mps / braking
    closing_speed = ego_speed_mps - lead_speed_mps
    collision_time_while_braking = (
        -closing_speed + sqrt(closing_speed**2 + 2.0 * braking * bumper_gap_m)
    ) / braking
    if collision_time_while_braking <= lead_stop_time:
        return max(collision_time_while_braking, 0.0)

    remaining_gap = (
        bumper_gap_m
        + stopping_distance(lead_speed_mps, lead_deceleration_mps2)
        - ego_speed_mps * lead_stop_time
    )
    return lead_stop_time + max(remaining_gap, 0.0) / ego_speed_mps


class HazardAgent:
    """Evaluate configured low-probability, high-severity alternatives."""

    agent_id = "hazard"

    def __init__(self, config: HazardAgentConfig) -> None:
        self._config = config

    def analyze(self, observation: SceneObservation) -> tuple[RiskClaim, ...]:
        candidates: list[_Candidate] = []
        for actor in sorted(observation.visible_actors, key=lambda item: item.actor_id):
            lead = self._lead_candidate(observation, actor)
            if lead is not None:
                candidates.append(lead)
            crossing = self._crossing_candidate(observation, actor)
            if crossing is not None:
                candidates.append(crossing)
        occlusion = self._occlusion_candidate(observation)
        if occlusion is not None:
            candidates.append(occlusion)
        if not candidates:
            return (neutral_claim(self.agent_id, observation),)
        return ordered_claims(candidate.claim for candidate in candidates)

    def _lead_candidate(
        self, observation: SceneObservation, actor: ActorState
    ) -> _Candidate | None:
        if not actor.same_lane or actor.relative_longitudinal_m <= 0.0:
            return None
        longitudinal_speed, _ = project_vector(actor.velocity_xy_mps, observation.ego.heading_rad)
        lead_speed = max(longitudinal_speed, 0.0)
        bumper_gap = actor.relative_longitudinal_m - 0.5 * (
            self._config.ego_length_m + actor.length_m
        )
        lead_stop_distance = stopping_distance(lead_speed, self._config.lead_max_deceleration_mps2)
        available_distance = bumper_gap + lead_stop_distance - self._config.safety_buffer_m
        margin = available_distance - self._ego_required_distance(observation)
        ttc = _lead_braking_ttc(
            bumper_gap_m=bumper_gap,
            ego_speed_mps=observation.ego.speed_mps,
            lead_speed_mps=lead_speed,
            lead_deceleration_mps2=self._config.lead_max_deceleration_mps2,
        )
        claim = self._finite_margin_claim(
            observation=observation,
            event_type="hazard_lead_braking",
            target_actor_id=actor.actor_id,
            margin_m=margin,
            available_distance_m=available_distance,
            ttc_s=ttc,
            evidence=(
                f"bumper_gap_m={bumper_gap:.6f}",
                f"lead_stop_distance_m={lead_stop_distance:.6f}",
            ),
            assumptions=(
                "lead_maximum_braking",
                "ego_constant_speed_for_ttc",
                "ego_reaction_then_safe_braking_for_margin",
            ),
        )
        return _Candidate(claim)

    def _crossing_candidate(
        self, observation: SceneObservation, actor: ActorState
    ) -> _Candidate | None:
        if actor.actor_type != "crossing_actor":
            return None
        conflict_distance = (
            observation.road_context.distance_to_conflict_point_m
            if observation.road_context.distance_to_conflict_point_m is not None
            else max(actor.relative_longitudinal_m, 0.0)
        )
        ego_arrival = self._ego_arrival_time(observation, conflict_distance)
        if ego_arrival is None:
            return None
        _, observed_lateral_speed = project_vector(
            actor.velocity_xy_mps, observation.ego.heading_rad
        )
        effective_crossing_speed = max(
            abs(observed_lateral_speed), self._config.crossing_actor_max_speed_mps
        )
        earliest_actor_arrival = abs(actor.relative_lateral_m) / effective_crossing_speed
        if earliest_actor_arrival > ego_arrival + self._config.crossing_occupancy_allowance_s:
            return None
        available_distance = conflict_distance - self._config.safety_buffer_m
        margin = available_distance - self._ego_required_distance(observation)
        claim = self._finite_margin_claim(
            observation=observation,
            event_type="hazard_crossing",
            target_actor_id=actor.actor_id,
            margin_m=margin,
            available_distance_m=available_distance,
            ttc_s=ego_arrival,
            evidence=(
                f"ego_arrival_s={ego_arrival:.6f}",
                f"earliest_actor_arrival_s={earliest_actor_arrival:.6f}",
            ),
            assumptions=("crossing_actor_can_delay_entry",),
        )
        return _Candidate(claim)

    def _occlusion_candidate(self, observation: SceneObservation) -> _Candidate | None:
        if not observation.occlusion_regions:
            return None
        conflict_distance = observation.road_context.distance_to_conflict_point_m
        if conflict_distance is None:
            event_type = "occlusion_hazard"
            claim = RiskClaim(
                claim_id=claim_id(self.agent_id, observation, event_type, None),
                agent_id=self.agent_id,
                event_type=event_type,
                target_actor_id=None,
                probability=None,
                confidence=0.7,
                severity=0.7,
                time_horizon_s=self._config.reaction_delay_s,
                min_ttc_s=None,
                stopping_margin_m=None,
                recommended_max_speed_mps=min(
                    observation.ego.speed_limit_mps,
                    self._config.occlusion_crawl_speed_mps,
                ),
                hard_stop_required=False,
                evidence=("occlusion_present",),
                assumptions=("conflict_distance_unavailable",),
                valid_until_step=observation.step_index,
            )
            return _Candidate(claim)
        ego_arrival = self._ego_arrival_time(observation, conflict_distance)
        available_distance = conflict_distance - self._config.safety_buffer_m
        margin = available_distance - self._ego_required_distance(observation)
        claim = self._finite_margin_claim(
            observation=observation,
            event_type="occlusion_hazard",
            target_actor_id=None,
            margin_m=margin,
            available_distance_m=available_distance,
            ttc_s=ego_arrival,
            evidence=(
                "occlusion_present",
                f"conflict_distance_m={conflict_distance:.6f}",
            ),
            assumptions=("virtual_actor_can_synchronize_entry",),
        )
        return _Candidate(claim)

    def _finite_margin_claim(
        self,
        *,
        observation: SceneObservation,
        event_type: str,
        target_actor_id: str | None,
        margin_m: float,
        available_distance_m: float,
        ttc_s: float | None,
        evidence: tuple[str, ...],
        assumptions: tuple[str, ...],
    ) -> RiskClaim:
        severity = self._margin_severity(margin_m)
        recommended_speed = min(
            observation.ego.speed_limit_mps,
            safe_speed_for_distance(
                available_distance_m,
                self._config.reaction_delay_s,
                self._config.ego_max_safe_deceleration_mps2,
            ),
        )
        return RiskClaim(
            claim_id=claim_id(self.agent_id, observation, event_type, target_actor_id),
            agent_id=self.agent_id,
            event_type=event_type,
            target_actor_id=target_actor_id,
            probability=None,
            confidence=0.85,
            severity=severity,
            time_horizon_s=max(ttc_s or 0.0, self._config.reaction_delay_s),
            min_ttc_s=ttc_s,
            stopping_margin_m=margin_m,
            recommended_max_speed_mps=recommended_speed,
            hard_stop_required=False,
            evidence=evidence,
            assumptions=assumptions,
            valid_until_step=observation.step_index,
        )

    def _ego_required_distance(self, observation: SceneObservation) -> float:
        return observation.ego.speed_mps * self._config.reaction_delay_s + stopping_distance(
            observation.ego.speed_mps,
            self._config.ego_max_safe_deceleration_mps2,
        )

    @staticmethod
    def _ego_arrival_time(
        observation: SceneObservation, conflict_distance_m: float
    ) -> float | None:
        if observation.ego.speed_mps <= 0.0:
            return None
        return max(conflict_distance_m, 0.0) / observation.ego.speed_mps

    def _margin_severity(self, margin_m: float) -> float:
        scaled = margin_m / self._config.severe_margin_scale_m
        if scaled >= 0.0:
            tail = exp(-scaled)
            return tail / (1.0 + tail)
        return 1.0 / (1.0 + exp(scaled))
