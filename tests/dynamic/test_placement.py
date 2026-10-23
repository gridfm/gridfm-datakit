"""Tests for gridfm_datakit.dynamic.placement."""

from pathlib import Path

import numpy as np
import pytest

from markers import needs_powsybl

from gridfm_datakit.dynamic.placement import (
    MAX_ANCHOR_ATTEMPTS,
    EventGraph,
    PlacementError,
    place_scenario,
)

pytestmark = needs_powsybl

PATH_NETWORK_IEEE14 = str(
    Path(__file__).parent
    / "dynawo/benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.iidm",
)

BUS2 = "_BUS____2_TN"
HOPS_FROM_BUS2 = {
    0: [2],
    1: [1, 3, 4, 5],
    2: [6, 7, 9],
    3: [8, 10, 11, 12, 13, 14],
}
TRANSFORMERS = {
    "_BUS____4-BUS____9-1_PT",
    "_BUS____4-BUS____7-1_PT",
    "_BUS____5-BUS____6-1_PT",
}
SEEDS = range(50)


def _bus(number):
    return f"_BUS__{number:_>3}_TN"


@pytest.fixture
def network():
    import pypowsybl.network as pn

    return pn.load(PATH_NETWORK_IEEE14)


@pytest.fixture
def graph(network):
    return EventGraph.from_network(network)


def _draw(graph, anchor, targets, seed):
    return place_scenario(graph, anchor, targets, np.random.default_rng(seed))


def test_distances_are_hop_counts(graph):
    expected = {
        _bus(number): hops
        for hops, numbers in HOPS_FROM_BUS2.items()
        for number in numbers
    }
    assert graph.distances(BUS2) == expected


def test_generators_one_hop_from_bus_2(graph):
    drawn = {_draw(graph, BUS2, [("generator", 1, 1)], s)[1][0] for s in SEEDS}
    assert drawn == {"_GEN____1_SM", "_GEN____3_SM"}


def test_lines_touching_bus_2(graph, network):
    lines = network.get_lines(
        attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id"],
    )
    touching = set(
        lines[
            (lines["bus_breaker_bus1_id"] == BUS2)
            | (lines["bus_breaker_bus2_id"] == BUS2)
        ].index,
    )
    drawn = {_draw(graph, BUS2, [("line", 0, 0)], s)[1][0] for s in SEEDS}
    assert drawn and drawn <= touching


def test_transformers_one_hop_from_bus_2(graph):
    drawn = {_draw(graph, BUS2, [("transformer", 1, 1)], s)[1][0] for s in SEEDS}
    assert drawn <= TRANSFORMERS and drawn


def test_targets_of_one_scenario_are_distinct(graph):
    for seed in SEEDS:
        _, (first, second) = _draw(
            graph,
            BUS2,
            [("generator", 0, 1), ("generator", 0, 1)],
            seed,
        )
        assert first != second


def test_a_wide_target_leaves_a_narrow_one_its_element(graph):
    for seed in SEEDS:
        _, (wide, narrow) = _draw(
            graph,
            BUS2,
            [("generator", 0, 1), ("generator", 0, 0)],
            seed,
        )
        assert narrow == "_GEN____2_SM"
        assert wide in {"_GEN____1_SM", "_GEN____3_SM"}


def test_excluded_elements_are_never_placed(graph):
    excluded = "_GEN____1_SM"
    targets = [("generator", 0, 1)]
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        _, (drawn,) = place_scenario(graph, BUS2, targets, rng, exclude=[excluded])
        assert drawn != excluded
        rng = np.random.default_rng(seed)
        unexcluded = place_scenario(graph, BUS2, targets, rng, exclude=())
        assert unexcluded == _draw(graph, BUS2, targets, seed)


def test_element_type_of_an_id(graph, network):
    assert graph.element_type("_GEN____2_SM") == "generator"
    assert graph.element_type(BUS2) == "bus"
    assert graph.element_type("_BUS____1-BUS____2-1_AC") == "line"
    assert graph.element_type("_GEN___99_SM") is None

    network.clone_variant("InitialState", "perturbed")
    network.set_working_variant("perturbed")
    network.update_generators(id="_GEN____3_SM", connected=False)
    assert EventGraph.from_network(network).element_type("_GEN____3_SM") is None


def test_random_anchor_is_reproducible_and_respects_distances(graph, network):
    targets = [("generator", 1, 2), ("load", 0, 1)]
    assert _draw(graph, None, targets, 7) == _draw(graph, None, targets, 7)

    generators = network.get_generators(attributes=["bus_breaker_bus_id"])
    loads = network.get_loads(attributes=["bus_breaker_bus_id"])
    bus_of = {
        **generators["bus_breaker_bus_id"].to_dict(),
        **loads["bus_breaker_bus_id"].to_dict(),
    }
    anchors = set()
    for seed in SEEDS:
        anchor, placed = _draw(graph, None, targets, seed)
        anchors.add(anchor)
        distance = graph.distances(anchor)
        for (_, low, high), element_id in zip(targets, placed):
            assert low <= distance[bus_of[element_id]] <= high
    assert len(anchors) > 1


def test_the_graph_sees_the_perturbed_variant(network):
    network.clone_variant("InitialState", "perturbed")
    network.set_working_variant("perturbed")
    network.update_branches(
        id=["_BUS____1-BUS____2-1_AC", "_BUS____1-BUS____5-1_AC"],
        connected1=[False, False],
        connected2=[False, False],
    )
    network.update_generators(id="_GEN____3_SM", connected=False)

    perturbed = EventGraph.from_network(network)
    assert _bus(1) not in perturbed.distances(BUS2)
    with pytest.raises(PlacementError, match=BUS2):
        _draw(perturbed, BUS2, [("generator", 1, 1)], 0)

    network.set_working_variant("InitialState")
    initial = EventGraph.from_network(network)
    assert _draw(initial, BUS2, [("generator", 1, 1)], 0)[0] == BUS2


B1 = "e44141af-f1dc-44d3-bfa4-b674e5c953d7"
B2 = "99b219f3-4593-428b-a4da-124a54630178"
B3 = "f96d552a-618d-4d0c-a39a-2dea3c411dee"
HUB = "5c74cb26-ce2f-40c6-951d-89091eb781b6"
LEAF = "a81d08ed-f51d-4538-8d1e-fb2d0dbd128e"
FAR = "f70f6bad-eb8d-4b8f-8431-4ab93581514e"
T3 = "84ed55f4-61f5-4d9d-8755-bba7b877a246"
G3 = "550ebe0d-f2b2-48c1-991f-cebea43a21aa"


@pytest.fixture
def micro_grid():
    import pypowsybl.network as pn

    return pn.create_micro_grid_be_network()


def test_three_winding_transformers_are_edges(micro_grid):
    graph = EventGraph.from_network(micro_grid)
    assert graph.distances(B2) == {B2: 0, HUB: 1, FAR: 1, B1: 1, B3: 1, LEAF: 2}
    assert {_draw(graph, B2, [("generator", 1, 1)], s)[1][0] for s in SEEDS} == {G3}


def test_a_three_winding_transformer_is_not_a_target(micro_grid):
    graph = EventGraph.from_network(micro_grid)
    assert graph.element_type(T3) is None
    assert all(
        T3 not in {element_id for element_id, _ in listed}
        for listed in graph.candidates.values()
    )


def test_a_disconnected_end_of_a_three_winding_transformer_is_not_an_edge(
    micro_grid,
):
    micro_grid.clone_variant(micro_grid.get_working_variant_id(), "perturbed")
    micro_grid.set_working_variant("perturbed")
    micro_grid.update_3_windings_transformers(id=T3, connected3=False)

    distances = EventGraph.from_network(micro_grid).distances(B2)
    assert distances[B1] == 1
    assert B3 not in distances


def test_an_unplaceable_random_scenario_gives_up(graph):
    with pytest.raises(PlacementError, match=str(MAX_ANCHOR_ATTEMPTS)):
        _draw(graph, None, [("generator", 20, 20)], 0)


@pytest.mark.parametrize(
    "anchor, targets",
    [
        ("_BUS___99_TN", [("generator", 0, 1)]),
        (BUS2, [("switch", 0, 0)]),
        (BUS2, [("load", 2, 1)]),
        (BUS2, [("load", -1, 1)]),
    ],
)
def test_invalid_requests_raise_value_error(graph, anchor, targets):
    with pytest.raises(ValueError):
        _draw(graph, anchor, targets, 0)


HOPS_FROM_S2 = {"S2VL1": 0, "S3VL1": 1, "S4VL1": 2}


@pytest.fixture
def node_breaker():
    import pypowsybl.network as pn

    return pn.create_four_substations_node_breaker_network()


def test_closed_switches_add_no_hop(node_breaker):
    buses = node_breaker.get_bus_breaker_view_buses(attributes=["voltage_level_id"])
    expected = {
        bus: HOPS_FROM_S2[level]
        for bus, level in buses["voltage_level_id"].items()
        if level in HOPS_FROM_S2
    }
    graph = EventGraph.from_network(node_breaker)
    assert graph.distances("S2VL1_0") == expected
    assert _draw(graph, "S2VL1_0", [("load", 2, 2)], 0)[1] == ["LD6"]


def test_an_open_switch_is_not_an_edge(node_breaker):
    node_breaker.clone_variant(node_breaker.get_working_variant_id(), "perturbed")
    node_breaker.set_working_variant("perturbed")
    node_breaker.update_switches(id="S4VL1_LD6_BREAKER", open=True)

    graph = EventGraph.from_network(node_breaker)
    assert "S4VL1_2" not in graph.distances("S2VL1_0")
    with pytest.raises(PlacementError):
        _draw(graph, "S2VL1_0", [("load", 2, 2)], 0)
