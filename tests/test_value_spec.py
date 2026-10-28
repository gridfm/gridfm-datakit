"""Tests for gridfm_datakit.utils.value_spec."""

import math
import re

import numpy as np
import pytest

from gridfm_datakit.utils.value_spec import (
    NON_NEGATIVE,
    POSITIVE,
    Choice,
    Fixed,
    Normal,
    Uniform,
    parse_value_spec,
)

PATH = "event_perturbation.scenarios[0].start_time"


def _draws(spec, seed, n=10_000):
    rng = np.random.default_rng(seed)
    return np.array([spec.sample(rng) for _ in range(n)])


class TestParse:
    def test_a_number_is_fixed(self):
        spec = parse_value_spec(3, PATH)
        assert spec == Fixed(3.0)
        assert spec.support() == (3.0, 3.0)
        assert spec.sample(np.random.default_rng(0)) == 3.0

    def test_normal(self):
        spec = parse_value_spec({"distribution": "normal", "mean": 1, "std": 2}, PATH)
        assert isinstance(spec, Normal)
        assert spec.support() == (-11.0, 13.0)

    def test_bounded_normal(self):
        spec = parse_value_spec(
            {"distribution": "normal", "mean": 0, "std": 1, "min": -0.5, "max": 2},
            PATH,
        )
        assert spec.support() == (-0.5, 2.0)

    def test_normal_with_a_floor(self):
        spec = parse_value_spec(
            {"distribution": "normal", "mean": 0.1, "std": 0.05},
            PATH,
            POSITIVE,
        )
        assert spec.support() == (0.0, pytest.approx(0.4))

    def test_one_sided_normal_support(self):
        spec = parse_value_spec(
            {"distribution": "normal", "mean": 10, "std": 1, "min": 15.9},
            PATH,
        )
        assert spec.support() == (15.9, pytest.approx(21.9))
        spec = parse_value_spec(
            {"distribution": "normal", "mean": 0, "std": 1, "max": -5.9},
            PATH,
        )
        assert spec.support() == (pytest.approx(-11.9), -5.9)

    def test_uniform(self):
        spec = parse_value_spec({"distribution": "uniform", "low": 1, "high": 4}, PATH)
        assert isinstance(spec, Uniform)
        assert spec.support() == (1.0, 4.0)

    def test_choice(self):
        spec = parse_value_spec(
            {"distribution": "choice", "values": [3, 1, 2], "weights": [1, 1, 2]},
            PATH,
        )
        assert isinstance(spec, Choice)
        assert spec.support() == (1, 3)


class TestRejects:
    def _raises(self, spec, *names, floor=None):
        with pytest.raises(ValueError) as info:
            parse_value_spec(spec, PATH, floor)
        message = str(info.value)
        assert message.startswith(f"{PATH}: "), message
        for name in names:
            assert name in message, message

    def test_sigma_is_not_a_key(self):
        self._raises(
            {"distribution": "normal", "mean": 0, "std": 1, "sigma": 1},
            "sigma",
        )

    def test_missing_distribution(self):
        self._raises({"mean": 0, "std": 1}, "normal", "uniform", "choice")

    def test_a_list_distribution(self):
        self._raises(
            {"distribution": ["uniform"], "low": 1, "high": 2},
            "normal",
            "uniform",
            "choice",
        )

    def test_unknown_distribution(self):
        self._raises(
            {"distribution": "lognormal", "mean": 0, "std": 1},
            "lognormal",
            "normal",
            "uniform",
            "choice",
        )

    @pytest.mark.parametrize(
        "spec",
        [
            {"distribution": "normal", "mean": 0, "std": 0},
            {"distribution": "uniform", "low": 1, "high": 1},
            {"distribution": "normal", "mean": 0, "std": 1, "min": 1, "max": -1},
            {"distribution": "choice", "values": []},
            {"distribution": "choice", "values": [1, 2], "weights": [1]},
            {"distribution": "choice", "values": [1, 2], "weights": [1, -1]},
            {"distribution": "choice", "values": [1, 2], "weights": [0, 0]},
            {"distribution": "normal", "mean": True, "std": 1},
            True,
        ],
    )
    def test_invalid_parameters(self, spec):
        self._raises(spec)

    def test_missing_parameter(self):
        self._raises({"distribution": "uniform", "low": 0}, "high")

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_numbers(self, value):
        self._raises(value, "finite")
        self._raises({"distribution": "normal", "mean": 0, "std": value}, "std")


class TestSample:
    def test_bounded_normal_redraws_rather_than_clips(self):
        spec = parse_value_spec(
            {"distribution": "normal", "mean": 0, "std": 1, "min": -0.5, "max": 0.5},
            PATH,
        )
        draws = _draws(spec, 0)
        assert draws.min() > -0.5 and draws.max() < 0.5
        assert np.array_equal(draws, _draws(spec, 0))
        assert not np.array_equal(draws, _draws(spec, 1))

    @pytest.mark.parametrize(
        "spec",
        [
            {"mean": 0, "std": 1, "min": 5.9},
            {"mean": 0, "std": 1, "max": -5.9},
            {"mean": 10, "std": 1, "min": 15.9},
        ],
    )
    def test_one_sided_tail_draws_stay_inside_the_support(self, spec):
        spec = parse_value_spec({"distribution": "normal", **spec}, PATH)
        low, high = spec.support()
        draws = _draws(spec, 0)
        assert ((low <= draws) & (draws <= high)).all()

    def test_an_unbounded_normal_is_drawn_inside_its_support(self, monkeypatch):
        calls = []

        class Stub:
            def rvs(self, a, b, **kwargs):
                calls.append((a, b))
                return 0.0

        monkeypatch.setattr("gridfm_datakit.utils.value_spec.truncnorm", Stub())
        spec = parse_value_spec({"distribution": "normal", "mean": 2, "std": 1}, PATH)
        spec.sample(np.random.default_rng(0))
        assert calls == [(-6.0, 6.0)]

    def test_normal_draws_are_floats(self):
        spec = parse_value_spec({"distribution": "normal", "mean": 0, "std": 1}, PATH)
        assert type(spec.sample(np.random.default_rng(0))) is float

    def test_uniform_stays_in_its_half_open_interval(self):
        spec = parse_value_spec({"distribution": "uniform", "low": 2, "high": 3}, PATH)
        draws = _draws(spec, 0)
        assert draws.min() >= 2 and draws.max() < 3
        assert type(spec.sample(np.random.default_rng(0))) is float

    def test_choice_never_draws_a_zero_weight(self):
        spec = parse_value_spec(
            {
                "distribution": "choice",
                "values": ["a", "b", "c"],
                "weights": [1, 0, 3],
            },
            PATH,
        )
        rng = np.random.default_rng(0)
        draws = {spec.sample(rng) for _ in range(1_000)}
        assert draws == {"a", "c"}


class TestFloor:
    def test_fixed_zero(self):
        with pytest.raises(ValueError, match=f"^{re.escape(PATH)}: "):
            parse_value_spec(0, PATH, POSITIVE)
        assert parse_value_spec(0, PATH, NON_NEGATIVE) == Fixed(0.0)

    def test_uniform_low_must_be_admitted(self):
        with pytest.raises(ValueError, match=f"^{re.escape(PATH)}: "):
            parse_value_spec(
                {"distribution": "uniform", "low": -1, "high": 1},
                PATH,
                POSITIVE,
            )

    def test_normal_is_truncated_at_the_floor(self):
        spec = parse_value_spec(
            {"distribution": "normal", "mean": 0.1, "std": 0.05},
            PATH,
            POSITIVE,
        )
        assert (_draws(spec, 0) > 0).all()

    def test_normal_redraws_a_value_on_a_strict_floor(self, monkeypatch):
        class Stub:
            def __init__(self):
                self.values = iter([0.0, 0.3])

            def rvs(self, *args, **kwargs):
                return next(self.values)

        spec = {"distribution": "normal", "mean": 0.1, "std": 0.05}
        rng = np.random.default_rng(0)
        monkeypatch.setattr("gridfm_datakit.utils.value_spec.truncnorm", Stub())
        positive = parse_value_spec(spec, PATH, POSITIVE)
        assert positive.strict
        assert positive.sample(rng) == 0.3
        monkeypatch.setattr("gridfm_datakit.utils.value_spec.truncnorm", Stub())
        assert parse_value_spec(spec, PATH, NON_NEGATIVE).sample(rng) == 0.0
        assert not parse_value_spec({**spec, "min": 0.05}, PATH, POSITIVE).strict

    def test_a_strict_floor_too_close_to_max_raises(self):
        spec = {"distribution": "normal", "mean": 0, "std": 1, "max": 1e-200}
        positive = parse_value_spec(spec, PATH, POSITIVE)
        with pytest.raises(ValueError, match="no draw above 0.0"):
            positive.sample(np.random.default_rng(0))

    def test_normal_far_below_the_floor(self):
        with pytest.raises(ValueError, match=f"^{re.escape(PATH)}: "):
            parse_value_spec(
                {"distribution": "normal", "mean": -10, "std": 1},
                PATH,
                POSITIVE,
            )

    def test_choice_of_a_string(self):
        with pytest.raises(ValueError, match=f"^{re.escape(PATH)}: "):
            parse_value_spec(
                {"distribution": "choice", "values": [1, "a"]},
                PATH,
                POSITIVE,
            )

    def test_admits(self):
        assert not POSITIVE.admits(0.0) and POSITIVE.admits(math.ulp(0.0))
        assert NON_NEGATIVE.admits(0.0) and not NON_NEGATIVE.admits(-1e-12)
