"""Tests for gridfm_datakit.dynamic.event_perturbation."""

import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from markers import needs_powsybl

from gridfm_datakit.dynamic.dynawo.utils import EVENT_PARAMS_MAPPING
from gridfm_datakit.dynamic.event_perturbation import (
    EVENT_RECORD_COLUMNS,
    EventPerturbation,
    EventTarget,
    FixedTarget,
    check_event_perturbation,
    draw_events,
    parse_event_perturbation,
)
from gridfm_datakit.dynamic.placement import EventGraph, PlacementError
from gridfm_datakit.utils.param_handler import NestedNamespace

PATH = "dynamic.event_perturbation"
SCENARIO = f"{PATH}.scenarios[0]"
EVENT = f"{SCENARIO}.events[0]"
ELEMENTS = {
    "NodeFault": "bus",
    "ActivePowerVariation": "load",
    "ReactivePowerVariation": "generator",
    "ReferenceVoltageVariation": "generator",
}
PATH_NETWORK_IEEE14 = str(
    Path(__file__).parent
    / "dynawo/benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.iidm",
)


def _random(**overrides):
    block = {
        "type": "random",
        "n_event_variants": 3,
        "scenarios": [
            {
                "name": "generator_trip",
                "anchor": "random",
                "start_time": {"distribution": "uniform", "low": 20, "high": 80},
                "events": [
                    {
                        "type": "Disconnect",
                        "target": {"element": "generator", "distance": [0, 2]},
                    },
                ],
            },
        ],
    }
    block.update(overrides)
    return block


def _parse(block, start_time=0.0, stop_time=100.0):
    return parse_event_perturbation(block, start_time, stop_time)


def _raises(block, path, start_time=0.0, stop_time=100.0):
    with pytest.raises(ValueError, match=f"^{re.escape(path)}"):
        _parse(block, start_time, stop_time)


def _edit(edit):
    block = _random()
    edit(block)
    return block


def _event(kind, element, params=None, start_time=None):
    def edit(block):
        event = {"type": kind, "target": {"element": element, "distance": [0, 2]}}
        if params is not None:
            event["params"] = params
        block["scenarios"][0]["events"] = [event]
        if start_time is not None:
            block["scenarios"][0]["start_time"] = start_time

    return _edit(edit)


def _with_second_event(extra):
    def edit(block):
        block["scenarios"][0]["events"].append(
            {
                "type": "NodeFault",
                "target": {"element": "bus", "distance": [0, 2]},
                "params": {"fault_time": 0.1, "r_pu": 0, "x_pu": 0.2},
                **extra,
            },
        )

    return _edit(edit)


def _delayed_fault(delay):
    block = _fault()
    block["scenarios"][0]["start_time"] = {
        "distribution": "uniform",
        "low": 20,
        "high": 99.5,
    }
    block["scenarios"][0]["events"][0]["delay"] = delay
    return block


def _fixed_events(*events):
    def edit(block):
        block["scenarios"][0]["events"] = list(events)

    return _edit(edit)


def _trip(target):
    return {"type": "Disconnect", "target": target}


def _fault(**params):
    return _event(
        "NodeFault",
        "bus",
        {"fault_time": 0.1, "r_pu": 0, "x_pu": 0.2, **params},
    )


class TestParse:
    def test_absent_block_is_file(self):
        assert _parse(None) == EventPerturbation()
        assert EventPerturbation().type == "file"

    @pytest.mark.parametrize("kind", ["none", "file"])
    def test_none_and_file(self, kind):
        assert _parse({"type": kind}) == EventPerturbation(kind, 1, ())

    def test_random(self):
        perturbation = _parse(_random())
        assert perturbation.type == "random"
        assert perturbation.n_event_variants == 3
        (scenario,) = perturbation.scenarios
        assert scenario.name == "generator_trip"
        assert scenario.anchor is None
        assert scenario.start_time.support() == (20.0, 80.0)
        (event,) = scenario.events
        assert event.type == "Disconnect"
        assert event.target == EventTarget("generator", 0, 2)

    def test_namespace_block_is_not_mutated(self):
        block = NestedNamespace(**_random())
        before = copy.deepcopy(block.to_dict())
        _parse(block)
        assert block.to_dict() == before

    def test_a_fixed_target(self):
        fixed = {"static_id": "_GEN____2_SM"}
        (event,) = _parse(_fixed_events(_trip(fixed))).scenarios[0].events
        assert event.target == FixedTarget("_GEN____2_SM")
        fault = {
            "type": "NodeFault",
            "target": fixed,
            "params": {"fault_time": 0.1, "r_pu": 0, "x_pu": 0.2},
        }
        (event,) = _parse(_fixed_events(fault)).scenarios[0].events
        assert event.target == FixedTarget("_GEN____2_SM")

    def test_fixed_anchor_and_single_distance(self):
        def edit(block):
            block["scenarios"][0]["anchor"] = "_BUS____2_TN"
            block["scenarios"][0]["events"][0]["target"]["distance"] = 1

        scenario = _parse(_edit(edit)).scenarios[0]
        assert scenario.anchor == "_BUS____2_TN"
        assert scenario.events[0].target == EventTarget("generator", 1, 1)

    @pytest.mark.parametrize(
        "kind",
        sorted(set(EVENT_PARAMS_MAPPING) - {"Disconnect"}),
    )
    def test_every_event_type_parses_its_params(self, kind):
        params = {key: 0.1 for key in EVENT_PARAMS_MAPPING[kind]}
        (event,) = _parse(_event(kind, ELEMENTS[kind], params)).scenarios[0].events
        assert event.type == kind
        assert [key for key, _ in event.params] == EVENT_PARAMS_MAPPING[kind]

    def test_a_negative_delta(self):
        event = (
            _parse(
                _event("ActivePowerVariation", "load", {"delta_p": -0.5}),
            )
            .scenarios[0]
            .events[0]
        )
        assert event.params[0][1].support() == (-0.5, -0.5)

    def test_a_second_scenario(self):
        def edit(block):
            second = copy.deepcopy(block["scenarios"][0])
            second["name"] = "another_trip"
            block["scenarios"].append(second)

        scenarios = _parse(_edit(edit)).scenarios
        assert [s.name for s in scenarios] == ["generator_trip", "another_trip"]
        assert [s.weight for s in scenarios] == [1.0, 1.0]

    def test_a_second_event(self):
        delay = {"distribution": "uniform", "low": 0.1, "high": 0.3}
        scenario = _parse(_with_second_event({"delay": delay})).scenarios[0]
        first, second = scenario.events
        assert first.delay.support() == (0.0, 0.0)
        assert second.delay.support() == (0.1, 0.3)

    def test_a_delayed_fault_ending_before_the_stop_time(self):
        _parse(_delayed_fault(0.3))

    def test_a_fault_ending_before_the_stop_time(self):
        start = {"distribution": "uniform", "low": 20, "high": 99.8}
        _parse(
            _event(
                "NodeFault",
                "bus",
                {"fault_time": 0.1, "r_pu": 0, "x_pu": 0.2},
                start,
            ),
        )


class TestReject:
    @pytest.mark.parametrize("kind", ["none", "file"])
    def test_event_variants_outside_random(self, kind):
        _raises({"type": kind, "n_event_variants": 2}, f"{PATH}.n_event_variants")

    @pytest.mark.parametrize("value", [0, True, -1, 1.5])
    def test_invalid_event_variant_count(self, value):
        _raises(_random(n_event_variants=value), f"{PATH}.n_event_variants")

    def test_missing_type(self):
        _raises({"n_event_variants": 1}, f"{PATH}.type")

    def test_unknown_type(self):
        _raises({"type": "csv"}, f"{PATH}.type")

    def test_scenarios_outside_random(self):
        _raises({"type": "file", "scenarios": []}, f"{PATH}.scenarios")

    def test_random_needs_the_solver_window(self):
        _raises(_random(), PATH, start_time=None, stop_time=100.0)
        _raises(_random(), PATH, start_time=0.0, stop_time=None)

    @pytest.mark.parametrize(
        "edit, path",
        [
            (lambda b: b.update(seed=1), f"{PATH}.seed"),
            (
                lambda b: b["scenarios"][0].update(probability=1),
                f"{SCENARIO}.probability",
            ),
            (
                lambda b: b["scenarios"][0]["events"][0].update(offset=1),
                f"{SCENARIO}.events[0].offset",
            ),
            (
                lambda b: b["scenarios"][0]["events"][0]["target"].update(
                    static_id="_GEN____1_SM",
                ),
                f"{SCENARIO}.events[0].target.distance",
            ),
        ],
    )
    def test_unknown_keys_name_their_path(self, edit, path):
        with pytest.raises(ValueError, match=f"^{re.escape(path)}: unknown key"):
            _parse(_edit(edit))

    def test_no_scenario(self):
        _raises(_random(scenarios=[]), f"{PATH}.scenarios")

    @pytest.mark.parametrize("weight", [0, -1, True, "1"])
    def test_invalid_weight(self, weight):
        _raises(
            _edit(lambda b: b["scenarios"][0].update(weight=weight)),
            f"{SCENARIO}.weight",
        )

    @pytest.mark.parametrize("static_id", ["", 2])
    def test_an_invalid_static_id(self, static_id):
        _raises(
            _fixed_events(_trip({"static_id": static_id})),
            f"{EVENT}.target.static_id",
        )

    def test_a_repeated_fixed_target(self):
        trip = _trip({"static_id": "_GEN____2_SM"})
        _raises(
            _fixed_events(trip, copy.deepcopy(trip)),
            f"{SCENARIO}.events[1].target.static_id",
        )

    def test_a_repeated_scenario_name(self):
        _raises(
            _edit(lambda b: b["scenarios"].append(copy.deepcopy(b["scenarios"][0]))),
            f"{PATH}.scenarios[1].name",
        )

    def test_no_event(self):
        _raises(
            _edit(lambda b: b["scenarios"][0].update(events=[])),
            f"{SCENARIO}.events",
        )

    def test_a_negative_delay(self):
        _raises(
            _edit(lambda b: b["scenarios"][0]["events"][0].update(delay=-0.1)),
            f"{EVENT}.delay",
        )

    def test_a_delay_past_the_stop_time(self):
        block = _with_second_event({"delay": 0.1})
        block["scenarios"][0]["start_time"]["high"] = 99.95
        _raises(block, f"{SCENARIO}.events[1].delay")

    def test_a_delayed_fault_ending_after_the_stop_time(self):
        _raises(_delayed_fault(0.45), f"{EVENT}.params.fault_time")

    @pytest.mark.parametrize("kind", ["Trip", ["Disconnect"]])
    def test_another_event_type(self, kind):
        _raises(
            _edit(lambda b: b["scenarios"][0]["events"][0].update(type=kind)),
            f"{SCENARIO}.events[0].type",
        )

    @pytest.mark.parametrize(
        "kind, element",
        [
            ("NodeFault", "generator"),
            ("ReferenceVoltageVariation", "load"),
            ("ActivePowerVariation", "line"),
        ],
    )
    def test_an_element_the_type_does_not_accept(self, kind, element):
        params = {key: 0.1 for key in EVENT_PARAMS_MAPPING[kind]}
        _raises(_event(kind, element, params), f"{EVENT}.target.element")

    def test_a_missing_param(self):
        block = _event("NodeFault", "bus", {"fault_time": 0.1, "x_pu": 0.2})
        with pytest.raises(
            ValueError,
            match=f"^{re.escape(EVENT)}.params.r_pu: required key is missing",
        ):
            _parse(block)

    def test_a_param_of_another_type(self):
        with pytest.raises(
            ValueError,
            match=f"^{re.escape(EVENT)}.params.delta_p: unknown key",
        ):
            _parse(_fault(delta_p=0.1))

    def test_params_on_a_disconnection(self):
        _raises(
            _edit(
                lambda b: b["scenarios"][0]["events"][0].update(
                    params={"disconnect_only": 0},
                ),
            ),
            f"{EVENT}.params",
        )

    def test_missing_params(self):
        with pytest.raises(
            ValueError,
            match=f"^{re.escape(EVENT)}.params: required key is missing",
        ):
            _parse(_event("ActivePowerVariation", "load"))

    @pytest.mark.parametrize(
        "key, spec",
        [
            ("fault_time", 0),
            ("r_pu", -0.1),
            ("x_pu", {"distribution": "uniform", "low": -1, "high": 1}),
        ],
    )
    def test_a_param_below_its_floor(self, key, spec):
        _raises(_fault(**{key: spec}), f"{EVENT}.params.{key}")

    def test_a_delta_that_is_not_a_number(self):
        _raises(
            _event(
                "ActivePowerVariation",
                "load",
                {"delta_p": {"distribution": "choice", "values": ["a"]}},
            ),
            f"{EVENT}.params.delta_p",
        )

    def test_a_fault_ending_after_the_stop_time(self):
        start = {"distribution": "uniform", "low": 20, "high": 99.95}
        _raises(
            _event(
                "NodeFault",
                "bus",
                {"fault_time": 0.1, "r_pu": 0, "x_pu": 0.2},
                start,
            ),
            f"{EVENT}.params.fault_time",
        )

    def test_unknown_element(self):
        _raises(
            _edit(
                lambda b: b["scenarios"][0]["events"][0]["target"].update(
                    element="switch",
                ),
            ),
            f"{SCENARIO}.events[0].target.element",
        )

    @pytest.mark.parametrize("distance", [-1, [2, 1], True, [0, True], [1], "1"])
    def test_invalid_distance(self, distance):
        _raises(
            _edit(
                lambda b: b["scenarios"][0]["events"][0]["target"].update(
                    distance=distance,
                ),
            ),
            f"{SCENARIO}.events[0].target.distance",
        )

    def test_empty_name(self):
        _raises(_edit(lambda b: b["scenarios"][0].update(name="")), f"{SCENARIO}.name")

    @pytest.mark.parametrize(
        "spec",
        [{"distribution": "uniform", "low": 20, "high": 120}, -1],
    )
    def test_start_time_outside_the_window(self, spec):
        _raises(
            _edit(lambda b: b["scenarios"][0].update(start_time=spec)),
            f"{SCENARIO}.start_time",
        )

    def test_a_normal_start_time_past_the_stop_time(self):
        _raises(
            _edit(
                lambda b: b["scenarios"][0].update(
                    start_time={
                        "distribution": "normal",
                        "mean": 10,
                        "std": 1,
                        "min": 15.9,
                    },
                ),
            ),
            f"{SCENARIO}.start_time",
            stop_time=16.0,
        )


def _chain_graph():
    buses = [f"B{i}" for i in range(6)]
    adjacency = {
        bus: tuple(other for other in buses if abs(int(other[1:]) - int(bus[1:])) == 1)
        for bus in buses
    }
    return EventGraph(
        adjacency=adjacency,
        candidates={
            "bus": tuple((bus, (bus,)) for bus in buses),
            "generator": tuple((f"G{i}", (f"B{i}",)) for i in range(6)),
            "load": (),
            "line": tuple((f"L{i}", (f"B{i}", f"B{i + 1}")) for i in range(5)),
            "transformer": (),
        },
    )


class TestDrawEvents:
    def test_columns_and_values(self):
        graph = _chain_graph()
        perturbation = _parse(_random())
        frame = draw_events(perturbation, graph, np.random.default_rng([1, 0, 0, 0]))
        assert list(frame.columns) == EVENT_RECORD_COLUMNS
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["scenario"] == "generator_trip"
        assert row["event_name"] == "Disconnect"
        assert row["params"] == "disconnect_only=;"
        assert frame["start_time"].dtype == float

    def test_draws_are_reproducible_and_bounded(self):
        graph = _chain_graph()
        perturbation = _parse(
            _edit(lambda b: b["scenarios"][0].update(anchor="B0")),
        )

        def draw(seed, event_index):
            rng = np.random.default_rng([seed, 0, 0, event_index])
            return draw_events(perturbation, graph, rng)

        differs = False
        for event_index in range(20):
            frame = draw(1, event_index)
            pd.testing.assert_frame_equal(frame, draw(1, event_index))
            differs |= not frame.equals(draw(2, event_index))
            start = frame["start_time"].iloc[0]
            assert 20 <= start < 80
            assert frame["static_id"].iloc[0] in {"G0", "G1", "G2"}
        assert differs

    def test_node_faults_are_reproducible_and_bounded(self):
        graph = _chain_graph()
        perturbation = _parse(
            _event(
                "NodeFault",
                "bus",
                {
                    "fault_time": {
                        "distribution": "normal",
                        "mean": 0.1,
                        "std": 0.02,
                        "min": 0.05,
                        "max": 0.2,
                    },
                    "r_pu": 0,
                    "x_pu": {"distribution": "uniform", "low": 0.01, "high": 0.1},
                },
            ),
        )

        def draw(event_index):
            rng = np.random.default_rng([1, 0, 0, event_index])
            return draw_events(perturbation, graph, rng)

        for event_index in range(20):
            frame = draw(event_index)
            pd.testing.assert_frame_equal(frame, draw(event_index))
            assert frame["event_name"].iloc[0] == "NodeFault"
            pairs = [item.split("=") for item in frame["params"].iloc[0].split(";")]
            assert [key for key, _ in pairs] == ["fault_time", "r_pu", "x_pu"]
            values = {key: float(value) for key, value in pairs}
            assert 0.05 <= values["fault_time"] <= 0.2
            assert dict(pairs)["r_pu"] == "0.0"
            assert 0.01 <= values["x_pu"] < 0.1

    def test_a_power_variation_draws_its_delta(self):
        perturbation = _parse(
            _event("ActivePowerVariation", "generator", {"delta_p": 0.1}),
        )
        frame = draw_events(
            perturbation,
            _chain_graph(),
            np.random.default_rng([1, 0, 0, 0]),
        )
        assert frame["params"].iloc[0] == "delta_p=0.1"

    def test_a_trip_then_a_delayed_fault(self):
        graph = _chain_graph()
        perturbation = _parse(
            _with_second_event(
                {"delay": {"distribution": "uniform", "low": 0.1, "high": 0.3}},
            ),
        )

        def draw(event_index):
            rng = np.random.default_rng([1, 0, 0, event_index])
            return draw_events(perturbation, graph, rng)

        for event_index in range(20):
            frame = draw(event_index)
            pd.testing.assert_frame_equal(frame, draw(event_index))
            assert frame["event_name"].tolist() == ["Disconnect", "NodeFault"]
            start, later = frame["start_time"]
            assert 0.1 <= later - start < 0.3
            assert 20 <= start < 80

    def test_a_zero_delay_draws_as_no_delay(self):
        graph = _chain_graph()
        delayed = _parse(
            _edit(lambda b: b["scenarios"][0]["events"][0].update(delay=0)),
        )
        for event_index in range(20):
            frames = [
                draw_events(
                    perturbation,
                    graph,
                    np.random.default_rng([1, 0, 0, event_index]),
                )
                for perturbation in (_parse(_random()), delayed)
            ]
            pd.testing.assert_frame_equal(*frames)

    def test_scenarios_are_drawn_by_weight(self):
        graph = _chain_graph()
        trip = _random()["scenarios"][0]
        fault = _fault()["scenarios"][0]
        perturbation = _parse(
            _random(
                scenarios=[
                    {**trip, "name": "trip", "weight": 3},
                    {**fault, "name": "fault", "weight": 1},
                ],
            ),
        )
        kind = {"trip": "Disconnect", "fault": "NodeFault"}

        def draw(event_index):
            rng = np.random.default_rng([1, 0, 0, event_index])
            return draw_events(perturbation, graph, rng)

        trips = 0
        for event_index in range(2000):
            frame = draw(event_index)
            (name,) = frame["scenario"].unique()
            assert (frame["event_name"] == kind[name]).all()
            trips += name == "trip"
            pd.testing.assert_frame_equal(frame, draw(event_index))
        assert 0.7 <= trips / 2000 <= 0.8

    def test_a_fixed_target_beside_a_drawn_one(self):
        graph = _chain_graph()
        perturbation = _parse(
            _fixed_events(
                _trip({"static_id": "G2"}),
                _trip({"element": "generator", "distance": [0, 5]}),
            ),
        )

        def draw(event_index):
            rng = np.random.default_rng([1, 0, 0, event_index])
            return draw_events(perturbation, graph, rng)

        for event_index in range(50):
            frame = draw(event_index)
            pd.testing.assert_frame_equal(frame, draw(event_index))
            fixed, drawn = frame["static_id"]
            assert fixed == "G2"
            assert drawn != "G2"

    def test_a_scenario_of_fixed_targets_draws_only_its_time(self):
        graph = _chain_graph()
        perturbation = _parse(_fixed_events(_trip({"static_id": "L1"})))
        frames = [
            draw_events(perturbation, graph, np.random.default_rng([1, 0, 0, e]))
            for e in range(50)
        ]
        for frame in frames:
            pd.testing.assert_frame_equal(
                frame.drop(columns="start_time"),
                frames[0].drop(columns="start_time"),
            )
        assert frames[0]["static_id"].tolist() == ["L1"]

    def test_a_fixed_target_out_of_service_fails_the_variant(self):
        perturbation = _parse(_fixed_events(_trip({"static_id": "G9"})))
        with pytest.raises(PlacementError, match="G9"):
            draw_events(perturbation, _chain_graph(), np.random.default_rng(0))

    def test_no_scenario_draws_an_empty_frame(self):
        frame = draw_events(
            EventPerturbation(),
            _chain_graph(),
            np.random.default_rng(0),
        )
        assert frame.empty and list(frame.columns) == EVENT_RECORD_COLUMNS


@needs_powsybl
class TestCheck:
    @staticmethod
    def _fixed(anchor, distance):
        def edit(block):
            block["scenarios"][0]["anchor"] = anchor
            block["scenarios"][0]["events"][0]["target"]["distance"] = distance

        return _parse(_edit(edit))

    def test_a_placeable_fixed_anchor_passes(self):
        check_event_perturbation(self._fixed("_BUS____2_TN", 1), PATH_NETWORK_IEEE14)

    @pytest.mark.parametrize(
        "anchor, distance",
        [("_BUS____2_TN", 20), ("_BUS___99_TN", 1)],
    )
    def test_an_unusable_fixed_anchor_raises(self, anchor, distance):
        with pytest.raises(ValueError, match=f"^{re.escape(SCENARIO)}.anchor: "):
            check_event_perturbation(
                self._fixed(anchor, distance),
                PATH_NETWORK_IEEE14,
            )

    def test_a_fixed_target_of_the_network_passes(self):
        check_event_perturbation(
            _parse(_fixed_events(_trip({"static_id": "_GEN____2_SM"}))),
            PATH_NETWORK_IEEE14,
        )

    @pytest.mark.parametrize(
        "event",
        [
            _trip({"static_id": "_GEN___99_SM"}),
            {
                "type": "NodeFault",
                "target": {"static_id": "_GEN____2_SM"},
                "params": {"fault_time": 0.1, "r_pu": 0, "x_pu": 0.2},
            },
        ],
    )
    def test_an_unusable_fixed_target_raises(self, event):
        with pytest.raises(
            ValueError,
            match="^" + re.escape(f"{EVENT}.target.static_id: "),
        ):
            check_event_perturbation(_parse(_fixed_events(event)), PATH_NETWORK_IEEE14)

    def test_a_fixed_target_is_excluded_from_distance_targets(self):
        block = _fixed_events(
            _trip({"static_id": "_GEN____2_SM"}),
            _trip({"element": "generator", "distance": 0}),
        )
        block["scenarios"][0]["anchor"] = "_BUS____2_TN"
        with pytest.raises(ValueError, match=f"^{re.escape(SCENARIO)}.anchor: "):
            check_event_perturbation(_parse(block), PATH_NETWORK_IEEE14)

    def test_a_random_anchor_target_of_a_missing_type_raises(self, monkeypatch):
        monkeypatch.setattr(
            "gridfm_datakit.dynamic.event_perturbation.EventGraph.from_network",
            lambda pp_net: _chain_graph(),
        )
        perturbation = _parse(_event("ActivePowerVariation", "load", {"delta_p": 0.1}))
        with pytest.raises(
            ValueError,
            match="^" + re.escape(f"{EVENT}.target.element: "),
        ):
            check_event_perturbation(perturbation, PATH_NETWORK_IEEE14)

    def test_a_random_anchor_target_of_a_present_type_passes(self):
        check_event_perturbation(_parse(_random()), PATH_NETWORK_IEEE14)

    def test_file_mode_does_not_load_the_network(self, tmp_path):
        check_event_perturbation(
            EventPerturbation("file"),
            str(tmp_path / "missing.iidm"),
        )
