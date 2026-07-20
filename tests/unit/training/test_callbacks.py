import math

from mad_driving.training.callbacks import RewardComponentsCallback

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


class RecordingLogger:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.values[key] = value


class FakeModel:
    def __init__(self) -> None:
        self.logger = RecordingLogger()
        self.num_timesteps = 0


def initialized_callback() -> tuple[RewardComponentsCallback, FakeModel]:
    callback = RewardComponentsCallback()
    model = FakeModel()
    callback.init_callback(model)  # type: ignore[arg-type]
    return callback, model


def test_records_every_reward_component_from_vector_infos() -> None:
    callback, model = initialized_callback()
    first = {key: float(index) for index, key in enumerate(REWARD_COMPONENT_KEYS, start=1)}
    second = {key: value + 2.0 for key, value in first.items()}
    callback.locals = {
        "infos": [
            {"reward_components": first},
            {"reward_components": second},
        ]
    }

    assert callback._on_step() is True

    assert model.logger.values == {
        f"reward/{key}": (first[key] + second[key]) / 2.0 for key in REWARD_COMPONENT_KEYS
    }


def test_ignores_missing_and_malformed_reward_info_without_stopping_learning() -> None:
    callback, model = initialized_callback()
    callback.locals = {
        "infos": [
            {},
            None,
            {"reward_components": "not-a-mapping"},
            {"reward_components": {"progress_reward": math.inf}},
            {"reward_components": {"arrival_reward": True}},
            {"reward_components": {"unknown": 3.0}},
            {"reward_components": {"progress_reward": 1.25}},
        ]
    }

    assert callback._on_step() is True
    assert model.logger.values == {"reward/progress_reward": 1.25}


def test_ignores_non_sequence_infos() -> None:
    callback, model = initialized_callback()
    callback.locals = {"infos": {"reward_components": {"progress_reward": 1.0}}}

    assert callback._on_step() is True
    assert model.logger.values == {}
