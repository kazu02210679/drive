"""Validated configuration models."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveInt


class StrictFrozenModel(BaseModel):
    """Base class that rejects unknown settings and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MetaDriveConfig(StrictFrozenModel):
    """MetaDrive options used by the Phase 1 headless environment."""

    use_render: bool = False
    image_observation: bool = False
    num_scenarios: PositiveInt = 1
    start_seed: int = 0
    traffic_density: FiniteFloat = Field(default=0.1, ge=0.0, le=1.0)
    horizon: PositiveInt = 200


class AppConfig(StrictFrozenModel):
    """Root configuration for a deterministic Phase 1 smoke run."""

    seed: int = Field(ge=0)
    scenario_id: str = Field(min_length=1)
    decision_steps: PositiveInt
    fixed_action: tuple[FiniteFloat, FiniteFloat]
    metadrive: MetaDriveConfig

    def metadrive_dict(self) -> dict[str, Any]:
        """Return a plain dictionary accepted by MetaDrive."""

        return self.metadrive.model_dump()
