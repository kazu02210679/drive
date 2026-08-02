"""Deterministic traffic-rule and explicit-stop analysis."""

from collections.abc import Callable

from mad_driving.agents.claim_factory import claim_id
from mad_driving.config.models import RuleAgentConfig
from mad_driving.interfaces import RiskClaim, SceneSnapshot


class RuleAgent:
    """Apply explicit stop constraints before the ordinary speed limit."""

    agent_id = "rule"

    def __init__(self, config: RuleAgentConfig) -> None:
        del config

    def analyze(self, snapshot: SceneSnapshot) -> RiskClaim:
        stop_rules: tuple[tuple[Callable[[SceneSnapshot], bool], str, str], ...] = (
            (
                lambda scene: scene.collision_occurred,
                "collision_stop",
                "collision_occurred",
            ),
            (lambda scene: scene.off_road, "off_road_stop", "ego_off_road"),
            (
                lambda scene: scene.intersection_entry_prohibited,
                "intersection_stop",
                "intersection_entry_prohibited",
            ),
            (lambda scene: scene.stop_required, "scenario_stop", "stop_required"),
        )
        for applies, event_type, evidence in stop_rules:
            if applies(snapshot):
                return self._hard_stop_claim(snapshot, event_type, evidence)
        return self._speed_limit_claim(snapshot)

    def _hard_stop_claim(
        self, snapshot: SceneSnapshot, event_type: str, evidence: str
    ) -> RiskClaim:
        return RiskClaim(
            claim_id=claim_id(self.agent_id, snapshot, event_type, None),
            agent_id=self.agent_id,
            event_type=event_type,
            target_actor_id=None,
            probability=1.0,
            confidence=1.0,
            severity=1.0,
            time_horizon_s=0.0,
            min_ttc_s=0.0,
            stopping_margin_m=None,
            recommended_max_speed_mps=0.0,
            hard_stop_required=True,
            evidence=(evidence,),
            assumptions=(),
            valid_until_step=snapshot.step_index,
        )

    def _speed_limit_claim(self, snapshot: SceneSnapshot) -> RiskClaim:
        speed_mps = snapshot.ego.speed_mps
        limit_mps = snapshot.ego.speed_limit_mps
        overspeed_mps = max(speed_mps - limit_mps, 0.0)
        severity = min(overspeed_mps / max(limit_mps, 1.0), 1.0)
        probability = 1.0 if overspeed_mps > 0.0 else 0.0
        event_type = "speed_limit"
        return RiskClaim(
            claim_id=claim_id(self.agent_id, snapshot, event_type, None),
            agent_id=self.agent_id,
            event_type=event_type,
            target_actor_id=None,
            probability=probability,
            confidence=1.0,
            severity=severity,
            time_horizon_s=0.0,
            min_ttc_s=None,
            stopping_margin_m=None,
            recommended_max_speed_mps=limit_mps,
            hard_stop_required=False,
            evidence=("ego_speed", "speed_limit"),
            assumptions=(),
            valid_until_step=snapshot.step_index,
        )
