from mad_driving.scenarios import ScenarioParameterSampler


def test_sampler_repeats_identical_values() -> None:
    first = ScenarioParameterSampler(123).uniform("gap", 35.0, 55.0)
    second = ScenarioParameterSampler(123).uniform("gap", 35.0, 55.0)

    assert first == second


def test_sampler_chooses_from_the_given_stable_ordering() -> None:
    choice = ScenarioParameterSampler(8).choose(("nominal", "lead_brake"))

    assert choice in {"nominal", "lead_brake"}
