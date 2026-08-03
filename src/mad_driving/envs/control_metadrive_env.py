"""MetaDrive environment binding for the four-action control Policy."""

from typing import Any, cast

from mad_driving.config.models import ControlConfig
from mad_driving.control import LaneKeepingLongitudinalPolicy
from mad_driving.envs.multi_agent_speed_env import DrivingEnvironment


def create_control_metadrive_env(
    config: dict[str, object],
    control_config: ControlConfig,
) -> DrivingEnvironment:
    """Construct MetaDrive with the custom discrete lane-keeping Policy."""

    from metadrive import MetaDriveEnv  # type: ignore[import-untyped]

    class ControlMetaDriveEnv(MetaDriveEnv):  # type: ignore[misc]
        @classmethod
        def default_config(cls) -> Any:
            defaults = super().default_config()
            defaults.update(
                {
                    "agent_policy": LaneKeepingLongitudinalPolicy,
                    "control_config": control_config.model_dump(),
                },
                allow_add_new_key=True,
            )
            return defaults

    return cast(DrivingEnvironment, ControlMetaDriveEnv(config))
