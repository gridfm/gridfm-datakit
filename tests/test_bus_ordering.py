"""Continuous bus ids follow the bus-table order.

Sorting the original bus numbers used to assign id 0 to the smallest bus
number, so a case that does not list buses in increasing order had a row
that was not its id. Pd, bus type, and Vm live on the row. Generator buses
and branch ends live on the id. These tests pin the row and the id together.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit.network import Network, load_net_from_pglib, makeYbus
from gridfm_datakit.perturbations.admittance_perturbation import (
    NoAdmittancePerturbationGenerator,
    PerturbAdmittanceGenerator,
)
from gridfm_datakit.perturbations.generator_perturbation import (
    NoGenPerturbationGenerator,
    PermuteGenCostGenerator,
    PerturbGenCostGenerator,
)
from gridfm_datakit.perturbations.load_perturbation import (
    LoadScenariosFromAggProfile,
    Powergraph,
)
from gridfm_datakit.perturbations.topology_perturbation import (
    NMinusKGenerator,
    NoPerturbationGenerator,
    RandomComponentDropGenerator,
)
from gridfm_datakit.process.process_network import (
    apply_slack_single_gen,
    pf_post_processing,
    pf_preprocessing,
)
from gridfm_datakit.utils.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    T_BUS,
)
from gridfm_datakit.utils.idx_bus import (
    BASE_KV,
    BUS_I,
    BUS_TYPE,
    PD,
    PQ,
    PV,
    REF,
    VA,
    VM,
    VMAX,
    VMIN,
)
from gridfm_datakit.utils.idx_cost import POLYNOMIAL
from gridfm_datakit.utils.idx_gen import GEN_BUS, GEN_STATUS, PMAX, QMAX, QMIN, VG


def _network(buses, gen_buses, branches):
    """Build a network from ``(original_id, type, pd)`` rows, in that order."""
    bus = np.zeros((len(buses), 13))
    for row, (bus_i, bus_type, pd) in enumerate(buses):
        bus[row, BUS_I] = bus_i
        bus[row, BUS_TYPE] = bus_type
        bus[row, PD] = pd
        bus[row, VM] = 1.0
        bus[row, BASE_KV] = 100.0
        bus[row, VMAX] = 1.1
        bus[row, VMIN] = 0.9

    gen = np.zeros((len(gen_buses), 21))
    gen[:, GEN_BUS] = gen_buses
    gen[:, VG] = 1.0
    gen[:, GEN_STATUS] = 1
    gen[:, PMAX] = 200.0
    gen[:, QMAX] = 100.0
    gen[:, QMIN] = -100.0

    branch = np.zeros((len(branches), 13))
    for row, (f_bus, t_bus) in enumerate(branches):
        branch[row, F_BUS] = f_bus
        branch[row, T_BUS] = t_bus
        branch[row, BR_R] = 0.01
        branch[row, BR_X] = 0.1
        branch[row, BR_STATUS] = 1
        branch[row, ANGMIN] = -360.0
        branch[row, ANGMAX] = 360.0

    gencost = np.tile(
        np.array([POLYNOMIAL, 0.0, 0.0, 3.0, 0.01, 10.0, 0.0]),
        (len(gen_buses), 1),
    )
    return Network(
        {
            "version": "2",
            "baseMVA": 100.0,
            "bus": bus,
            "gen": gen,
            "branch": branch,
            "gencost": gencost,
        },
    )


def _unsorted():
    """Buses listed 5, 2, 9. Bus 5 is the reference and sits in row 0."""
    return _network(
        [(5, REF, 0.0), (2, PV, 50.0), (9, PQ, 80.0)],
        gen_buses=[5, 2],
        branches=[(5, 2), (2, 9)],
    )


def test_ids_follow_table_order_when_buses_are_unsorted():
    net = _unsorted()
    n = len(net.buses)

    assert np.array_equal(net.buses[:, BUS_I], np.arange(n))
    assert net.original_bus_indices.tolist() == [5, 2, 9]
    assert net.bus_index_mapping == {5: 0, 2: 1, 9: 2}
    assert net.reverse_bus_index_mapping == {0: 5, 1: 2, 2: 9}
    # Sorted numbering would have given 2→0, 5→1, 9→2.
    assert net.bus_index_mapping[5] != 1
    assert net.ref_bus_idx == 0
    assert int(net.buses[net.ref_bus_idx, BUS_TYPE]) == REF
    assert net.gens[:, GEN_BUS].tolist() == [0, 1]
    assert net.branches[:, F_BUS].tolist() == [0, 1]
    assert net.branches[:, T_BUS].tolist() == [1, 2]
    assert net.check_single_connected_component()


def test_sorted_file_matches_rank_numbering():
    """A case already listed in increasing order keeps the old ids."""
    net = _network(
        [(1, REF, 0.0), (2, PV, 10.0), (4, PQ, 20.0)],
        gen_buses=[1, 2],
        branches=[(1, 2), (2, 4)],
    )
    assert net.bus_index_mapping == {1: 0, 2: 1, 4: 2}
    assert np.array_equal(net.buses[:, BUS_I], np.arange(3))
    assert net.ref_bus_idx == 0


def test_duplicate_bus_numbers_are_rejected():
    bus = np.zeros((2, 13))
    bus[:, BUS_I] = [5, 5]
    bus[:, BUS_TYPE] = [REF, PQ]
    gen = np.zeros((1, 21))
    gen[0, GEN_BUS] = 5
    gen[0, GEN_STATUS] = 1
    branch = np.zeros((1, 13))
    gencost = np.array([[POLYNOMIAL, 0, 0, 3, 0, 0, 0]])
    with pytest.raises(ValueError, match="Duplicate bus numbers"):
        Network(
            {
                "version": "2",
                "baseMVA": 100.0,
                "bus": bus,
                "gen": gen,
                "branch": branch,
                "gencost": gencost,
            },
        )


def test_slack_balance_uses_the_reference_bus_when_unsorted():
    net = _unsorted()
    # In-service gens are [bus 5, bus 2]. Only the one on bus 5 is slack.
    pg_gen = np.array([30.0, 70.0])
    pg_bus = np.bincount(
        net.gens[net.idx_gens_in_service, GEN_BUS].astype(int),
        weights=pg_gen,
        minlength=len(net.buses),
    )
    pf = np.array([10.0, 0.0])
    pt = np.array([-10.0, 0.0])

    updated = apply_slack_single_gen(net, pg_gen, pg_bus, pf, pt)

    # balance = Pg(bus 5) - Pd(bus 5) - flow out of bus 5 = 30 - 0 - 10
    assert updated.tolist() == [10.0, 70.0]
    assert net.Pd[net.ref_bus_idx] == 0.0
    assert net.Pd[1] == 50.0


def test_only_the_non_slack_generator_is_droppable_when_unsorted():
    net = _unsorted()
    generator = RandomComponentDropGenerator(
        n_topology_variants=1,
        k=1,
        base_net=net,
        elements=["gen"],
    )
    # gen 0 is on the reference bus (5). gen 1 is on bus 2.
    assert generator.components_to_drop == [(1, "gen")]


def test_pv_bus_without_a_generator_is_demoted_when_unsorted():
    net = _network(
        [(5, PV, 0.0), (2, PV, 50.0), (9, REF, 0.0)],
        gen_buses=[5, 9],
        branches=[(5, 2), (2, 9)],
    )
    # Drop the reference generator. Bus 5 still has its generator.
    # Bus 2 is PV and never had one, so it becomes PQ.
    ref_gen = np.where(net.gens[:, GEN_BUS] == net.ref_bus_idx)[0]
    net.deactivate_gens(ref_gen)

    assert int(net.buses[0, BUS_TYPE]) == PV
    assert int(net.buses[1, BUS_TYPE]) == PQ
    assert int(net.buses[2, BUS_TYPE]) == REF


def test_opf_voltages_land_on_the_matching_row_when_unsorted():
    net = _unsorted()
    vm_by_original = {5: 1.11, 2: 0.95, 9: 1.05}
    va_by_original = {5: 0.0, 2: 0.2, 9: -0.4}
    res = {
        "solution": {
            "gen": {
                "1": {"pg": 0.3, "qg": 0.0},
                "2": {"pg": 0.7, "qg": 0.1},
            },
            "branch": {
                "1": {"pf": 0.1, "pt": -0.1},
                "2": {"pf": 0.2, "pt": -0.2},
            },
            "bus": {
                str(orig): {"vm": vm_by_original[orig], "va": va_by_original[orig]}
                for orig in (5, 2, 9)
            },
        },
    }

    pf_preprocessing(net, res)

    for row, orig in enumerate(net.original_bus_indices):
        assert net.buses[row, VM] == pytest.approx(vm_by_original[int(orig)])
        assert net.buses[row, VA] == pytest.approx(
            np.rad2deg(va_by_original[int(orig)]),
        )
    for gen_idx in net.idx_gens_in_service:
        row = int(net.gens[gen_idx, GEN_BUS])
        assert net.gens[gen_idx, VG] == pytest.approx(net.buses[row, VM])


def test_to_mpc_keeps_original_bus_numbers_in_file_order():
    net = _unsorted()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "case.m"
        net.to_mpc(str(path))
        text = path.read_text()
    start = text.index("mpc.bus = [")
    end = text.index("];", start)
    written = []
    for line in text[start:end].splitlines():
        line = line.strip().rstrip(";")
        if not line or line.startswith("%") or line.startswith("mpc"):
            continue
        written.append(int(float(line.split()[0])))
    assert written == [5, 2, 9]


def test_ybus_uses_the_row_as_the_bus_index_when_unsorted():
    net = _unsorted()
    ybus, _, _ = makeYbus(net.baseMVA, net.buses, net.branches)
    assert ybus.shape == (3, 3)
    # Branch 5-2 is row 0-1, branch 2-9 is row 1-2. Bus 9 has no direct
    # connection to bus 5, so the (0, 2) entry stays zero.
    assert ybus[0, 2] == 0
    assert ybus[0, 1] != 0
    assert ybus[1, 2] != 0


def test_pglib_case_ids_follow_its_bus_table():
    net = load_net_from_pglib("case14_ieee")
    n = len(net.buses)
    assert np.array_equal(net.buses[:, BUS_I], np.arange(n))
    for row, orig in enumerate(net.original_bus_indices):
        assert net.bus_index_mapping[int(orig)] == row
        assert net.reverse_bus_index_mapping[row] == int(orig)
    # case14 lists buses in increasing order, so this matches sorted numbering.
    assert list(net.original_bus_indices) == sorted(net.original_bus_indices.tolist())


# (original bus number, type, Pd). A cycle plus a chord so one branch can drop
# and the network stays in one piece. Generators sit on the reference bus and
# on every PV bus. The reference bus is not always the first row, and its load
# differs from row 0 so a slack balance that reads the wrong row fails.
_SHAPES = {
    "sorted-continuous-slack-first": [
        (1, REF, 5.0),
        (2, PV, 40.0),
        (3, PQ, 25.0),
        (4, PV, 10.0),
    ],
    "sorted-gaps-slack-middle": [
        (1, PV, 40.0),
        (2, REF, 15.0),
        (5, PQ, 25.0),
        (10, PV, 10.0),
    ],
    "unsorted-continuous-slack-last": [
        (4, PV, 40.0),
        (1, PQ, 25.0),
        (3, PV, 10.0),
        (2, REF, 12.0),
    ],
    "unsorted-gaps-slack-third": [
        (9, PQ, 25.0),
        (2, PV, 40.0),
        (11, REF, 8.0),
        (5, PV, 10.0),
    ],
}


def _meshed(spec):
    """Four buses with a cycle and one chord, in the given table order."""
    ids = [row[0] for row in spec]
    branches = [(ids[i], ids[(i + 1) % 4]) for i in range(4)]
    branches.append((ids[0], ids[2]))
    gens = [row[0] for row in spec if row[1] in (REF, PV)]
    return _network(spec, gens, branches)


def _assert_table_order(net, spec):
    """Bus, generator, and branch rows are unchanged, and the id is the row."""
    n = len(spec)
    assert np.array_equal(net.buses[:, BUS_I], np.arange(n))
    assert net.original_bus_indices.tolist() == [row[0] for row in spec]
    assert net.bus_index_mapping == {row[0]: i for i, row in enumerate(spec)}
    assert net.reverse_bus_index_mapping == {i: row[0] for i, row in enumerate(spec)}
    ref_row = next(i for i, row in enumerate(spec) if row[1] == REF)
    assert net.ref_bus_idx == ref_row
    assert int(net.buses[ref_row, BUS_TYPE]) == REF
    gen_rows = [i for i, row in enumerate(spec) if row[1] in (REF, PV)]
    assert net.gens[:, GEN_BUS].astype(int).tolist() == gen_rows
    expected_ends = [(i, (i + 1) % 4) for i in range(4)] + [(0, 2)]
    ends = list(
        zip(
            net.branches[:, F_BUS].astype(int),
            net.branches[:, T_BUS].astype(int),
        ),
    )
    assert ends == expected_ends
    assert np.allclose(net.buses[:, PD], [row[2] for row in spec])


def _assert_ybus_follows_rows(net):
    """In-service branches sit in Ybus at their row ids.

    ``makeYbus`` multiplies each branch by its status, so a dropped branch
    leaves a structural zero at that pair. The pair of rows with no branch
    stays zero as well.
    """
    ybus, _, _ = makeYbus(net.baseMVA, net.buses, net.branches)
    assert ybus.shape == (len(net.buses), len(net.buses))
    ends = list(
        zip(
            net.branches[:, F_BUS].astype(int),
            net.branches[:, T_BUS].astype(int),
        ),
    )
    in_service = {
        (min(f_bus, t_bus), max(f_bus, t_bus))
        for (f_bus, t_bus), status in zip(ends, net.branches[:, BR_STATUS])
        if status == 1
    }
    for f_bus, t_bus in ends:
        pair = (min(f_bus, t_bus), max(f_bus, t_bus))
        if pair in in_service:
            assert ybus[f_bus, t_bus] != 0
            assert ybus[t_bus, f_bus] != 0
        else:
            assert ybus[f_bus, t_bus] == 0
            assert ybus[t_bus, f_bus] == 0
    # No branch joins row 1 and row 3.
    assert ybus[1, 3] == 0
    assert ybus[3, 1] == 0


def _topology_generators(net):
    return [
        ("none", NoPerturbationGenerator()),
        ("n-1", NMinusKGenerator(k=1, base_net=net)),
        (
            "random-branch",
            RandomComponentDropGenerator(
                n_topology_variants=3,
                k=1,
                base_net=net,
                elements=["branch"],
            ),
        ),
        (
            "random-gen",
            RandomComponentDropGenerator(
                n_topology_variants=3,
                k=1,
                base_net=net,
                elements=["gen"],
            ),
        ),
        (
            "random-branch-and-gen",
            RandomComponentDropGenerator(
                n_topology_variants=3,
                k=1,
                base_net=net,
                elements=["branch", "gen"],
            ),
        ),
    ]


def _generation_generators(net):
    return [
        ("gen-none", NoGenPerturbationGenerator()),
        ("gen-permute", PermuteGenCostGenerator(net)),
        ("gen-scale", PerturbGenCostGenerator(net, sigma=0.2)),
    ]


def _admittance_generators(net):
    return [
        ("adm-none", NoAdmittancePerturbationGenerator()),
        ("adm-scale", PerturbAdmittanceGenerator(net, sigma=0.2)),
    ]


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_every_perturbation_keeps_element_order(shape):
    spec = _SHAPES[shape]
    ref_row = next(i for i, row in enumerate(spec) if row[1] == REF)
    base = _meshed(spec)
    _assert_table_order(base, spec)
    _assert_ybus_follows_rows(base)
    assert base.check_single_connected_component()

    buses_before = base.buses.copy()
    gens_before = base.gens.copy()
    branches_before = base.branches.copy()

    np.random.seed(0)
    n_checked = 0
    saw_branch_outage = False
    saw_gen_outage = False
    for topo_name, topo in _topology_generators(base):
        for gen_name, gen in _generation_generators(base):
            for adm_name, adm in _admittance_generators(base):
                produced = list(adm.generate(gen.generate(topo.generate(base))))
                assert produced, f"{shape}/{topo_name}/{gen_name}/{adm_name}"
                for net in produced:
                    _assert_table_order(net, spec)
                    _assert_ybus_follows_rows(net)
                    assert net.check_single_connected_component()
                    if np.any(net.branches[:, BR_STATUS] == 0):
                        saw_branch_outage = True
                    if np.any(net.gens[:, GEN_STATUS] == 0):
                        saw_gen_outage = True
                    n_checked += 1

    assert saw_branch_outage, shape
    assert saw_gen_outage, shape
    # 1 untouched + 6 N-1 + 3x3 random, each with 3 cost and 2 admittance modes.
    assert n_checked == (1 + 6 + 3 * 3) * 3 * 2
    # Generators copy the network. The base table is untouched.
    assert np.array_equal(base.buses, buses_before)
    assert np.array_equal(base.gens, gens_before)
    assert np.array_equal(base.branches, branches_before)

    # Drop each branch, and each non-slack generator, on its own copy.
    for branch_idx in range(base.branches.shape[0]):
        net = base.copy_for_perturbation()
        net.deactivate_branches(np.array([branch_idx]))
        _assert_table_order(net, spec)
        _assert_ybus_follows_rows(net)
        assert net.branches[branch_idx, BR_STATUS] == 0
        assert net.check_single_connected_component()

    non_slack_gens = np.flatnonzero(base.gens[:, GEN_BUS] != base.ref_bus_idx)
    assert non_slack_gens.size >= 1
    for gen_idx in non_slack_gens:
        net = base.copy_for_perturbation()
        net.deactivate_gens(np.array([gen_idx]))
        # Pd is unchanged. The PV bus that lost its generator becomes PQ.
        # The reference bus stays where it was.
        assert np.array_equal(net.buses[:, BUS_I], np.arange(len(spec)))
        assert net.ref_bus_idx == ref_row
        assert int(net.buses[ref_row, BUS_TYPE]) == REF
        assert net.gens[:, GEN_BUS].astype(int).tolist() == [
            i for i, row in enumerate(spec) if row[1] in (REF, PV)
        ]
        assert int(net.buses[net.gens[gen_idx, GEN_BUS].astype(int), BUS_TYPE]) == PQ
        _assert_ybus_follows_rows(net)


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_powergraph_load_stays_on_its_row(shape):
    spec = _SHAPES[shape]
    base = _meshed(spec)
    with tempfile.TemporaryDirectory() as tmp:
        scenarios = Powergraph("default")(
            base,
            n_scenarios=3,
            scenario_log=str(Path(tmp) / "log.txt"),
            max_iter=10,
            seed=7,
        )
    assert scenarios.shape == (len(spec), 3, 2)
    loaded = [row[2] for row in spec]
    for scenario in range(3):
        present = np.array(loaded) > 0
        if present.any():
            scale = scenarios[present, scenario, 0] / np.array(loaded)[present]
            assert np.allclose(scale, scale[0])
        assert np.allclose(scenarios[~present, scenario, 0], 0.0)
    copy = base.copy_for_perturbation()
    copy.Pd = scenarios[:, 0, 0]
    assert np.array_equal(copy.buses[:, BUS_I], np.arange(len(spec)))
    assert copy.original_bus_indices.tolist() == [row[0] for row in spec]
    _assert_ybus_follows_rows(copy)


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_slack_balance_follows_the_reference_row(shape):
    spec = _SHAPES[shape]
    net = _meshed(spec)
    ref = net.ref_bus_idx
    assert ref == next(i for i, row in enumerate(spec) if row[1] == REF)

    in_service = net.idx_gens_in_service
    pg_gen = np.full(len(in_service), 10.0)
    slack_gens = np.flatnonzero(net.gens[in_service, GEN_BUS] == ref)
    pg_gen[slack_gens[0]] = 30.0
    pg_bus = np.bincount(
        net.gens[in_service, GEN_BUS].astype(int),
        weights=pg_gen,
        minlength=len(net.buses),
    )
    pf = np.zeros(net.branches.shape[0])
    pt = np.zeros(net.branches.shape[0])
    touch = np.flatnonzero(
        (net.branches[:, F_BUS] == ref) | (net.branches[:, T_BUS] == ref),
    )[0]
    pf[touch] = 10.0
    pt[touch] = -10.0
    flows = 10.0 if int(net.branches[touch, F_BUS]) == ref else -10.0
    balance = 30.0 - spec[ref][2] - flows
    if ref != 0:
        assert 30.0 - spec[0][2] - flows != balance

    updated = apply_slack_single_gen(net, pg_gen, pg_bus, pf, pt)
    expected = pg_gen.copy()
    expected[slack_gens[0]] -= balance
    assert np.allclose(updated, expected)


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_agg_profile_load_stays_on_its_row(shape):
    spec = _SHAPES[shape]
    base = _meshed(spec)
    generator = LoadScenariosFromAggProfile(
        agg_load_name="default",
        sigma=0.0,
        change_reactive_power=False,
        global_range=0.0,
        max_scaling_factor=1.0,
        step_size=1.0,
        start_scaling_factor=1.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        scenarios = generator(
            base,
            n_scenarios=2,
            scenarios_log=str(Path(tmp) / "log.txt"),
            max_iter=50,
            seed=7,
        )
    assert scenarios.shape[0] == len(spec)
    loaded = np.array([row[2] for row in spec])
    present = loaded > 0
    for scenario in range(scenarios.shape[1]):
        scale = scenarios[present, scenario, 0] / loaded[present]
        assert np.allclose(scale, scale[0])
        assert np.allclose(scenarios[~present, scenario, 0], 0.0)
    copy = base.copy_for_perturbation()
    copy.Pd = scenarios[:, 0, 0]
    assert np.array_equal(copy.buses[:, BUS_I], np.arange(len(spec)))
    assert copy.gens[:, GEN_BUS].astype(int).tolist() == [
        i for i, row in enumerate(spec) if row[1] in (REF, PV)
    ]
    _assert_ybus_follows_rows(copy)


def _require(module):
    return pytest.importorskip(module)


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_powsybl_and_lightsim_keep_the_slack_row(shape):
    """Both solvers solve the reference bus, even when it is not row 0."""
    _require("pypowsybl")
    _require("lightsim2grid")
    import pypowsybl
    import gridfm_datakit.lightsim2grid as l2g
    from gridfm_datakit.powsybl.convert import to_powsybl
    from gridfm_datakit.powsybl.params import get_default_lf_params
    from gridfm_datakit.powsybl.preprocess import get_pf_res
    from gridfm_datakit.utils.column_names import BUS_COLUMNS

    spec = _SHAPES[shape]
    net = _meshed(spec)
    ref = net.ref_bus_idx
    assert ref == next(i for i, row in enumerate(spec) if row[1] == REF)

    conv = to_powsybl(net)
    # PowSyBl keeps the bus table order, original numbers included.
    assert list(conv.mapping_p2g.bus.values()) == list(range(len(spec)))
    assert [
        conv.mapping_p2g.reverse_bus_index_mapping[i] for i in range(len(spec))
    ] == [row[0] for row in spec]
    meta = pypowsybl.loadflow.run_ac(conv.pp_net, get_default_lf_params())
    assert meta[0].status_text == "Converged"
    slack_id = meta[0].slack_bus_results[0].id
    assert int(conv.mapping_p2g.bus[slack_id]) == ref

    res = get_pf_res(conv.pp_net, 0.0, meta, conv.mapping_p2g)
    # Keys are the original bus numbers, so a gap or a shuffle still lines up.
    assert set(res["solution"]["bus"]) == {str(row[0]) for row in spec}
    posted = pf_post_processing(0, net, res, None, False)
    bus = posted["bus"]
    c_bus = BUS_COLUMNS.index("bus")
    c_vm = BUS_COLUMNS.index("Vm")
    c_va = BUS_COLUMNS.index("Va")
    c_pg = BUS_COLUMNS.index("Pg")
    c_ref = BUS_COLUMNS.index("REF")
    assert bus[:, c_bus].tolist() == list(range(len(spec)))
    assert bus[ref, c_ref] == 1
    assert bus[:, c_ref].sum() == 1
    assert abs(bus[ref, c_va]) < 1e-3
    assert abs(bus[ref, c_pg]) > 1.0
    for row, (_, kind, _) in enumerate(spec):
        if kind == PQ:
            assert abs(bus[row, c_pg]) < 1e-6

    was = conv.pp_net.per_unit
    conv.pp_net.per_unit = True
    solved = conv.pp_net.get_buses()
    conv.pp_net.per_unit = was
    for pp_id, gfm_row in conv.mapping_p2g.bus.items():
        assert bus[int(gfm_row), c_vm] == pytest.approx(
            float(solved.loc[pp_id, "v_mag"]),
        )
        assert bus[int(gfm_row), c_va] == pytest.approx(
            float(np.rad2deg(solved.loc[pp_id, "v_angle"])),
        )

    converted = l2g.to_lightsim2grid(net)
    ls_res = l2g.run_ls_pf(
        converted.ls_net,
        net,
        converted.mapping_l2g,
        as_arrays=True,
    )
    _, gen_pq, vmva = ls_res["solution"]["arrays"]
    assert abs(vmva[ref, 1]) < 1e-6
    if ref != 0:
        assert abs(vmva[0, 1]) > 1e-4
        assert abs(bus[0, c_ref]) == 0
    y_ls = ls_res["solution"]["Ybus"]
    y_ref, _, _ = makeYbus(net.baseMVA, net.buses, net.branches)
    assert np.allclose(y_ls.toarray(), y_ref.toarray(), atol=1e-6)
    slack_gen = int(np.flatnonzero(net.gens[:, GEN_BUS] == ref)[0])
    assert abs(gen_pq[slack_gen, 0]) > 1e-3
    assert np.allclose(vmva[:, 0], bus[:, c_vm], atol=1e-5)


def test_dynawo_initial_state_keeps_the_slack_row():
    """Dynawo's balanced initial state uses the reference row, not row 0.

    The time-domain ``Simulation.run`` needs a native Dynawo install. Dynawo
    publishes Linux and Windows builds and points macOS at Docker, which is
    not available here. The indexing happens in this initial-state step.
    """
    _require("pypowsybl")
    from gridfm_datakit.dynamic.dynawo.simulate import (
        compute_balanced_static_state_dynawo,
    )
    from gridfm_datakit.generate import init_julia
    from gridfm_datakit.powsybl.convert import to_powsybl
    from gridfm_datakit.utils.column_names import BUS_COLUMNS

    julia = init_julia(50)
    saw_slack_off_first_row = False
    for spec in _SHAPES.values():
        net = _meshed(spec)
        ref = net.ref_bus_idx
        saw_slack_off_first_row = saw_slack_off_first_row or ref != 0
        conv = to_powsybl(net)
        pp_net, pf_data = compute_balanced_static_state_dynawo(
            conv.pp_net,
            net,
            julia,
            conv.mapping_p2g,
        )
        bus = np.asarray(pf_data["bus"])
        c_bus = BUS_COLUMNS.index("bus")
        c_ref = BUS_COLUMNS.index("REF")
        c_vm = BUS_COLUMNS.index("Vm")
        assert bus[:, c_bus].tolist() == list(range(len(spec)))
        assert bus[ref, c_ref] == 1
        assert bus[:, c_ref].sum() == 1
        was = pp_net.per_unit
        pp_net.per_unit = True
        solved = pp_net.get_buses()["v_mag"]
        pp_net.per_unit = was
        for pp_id, gfm_row in conv.mapping_p2g.bus.items():
            assert bus[int(gfm_row), c_vm] == pytest.approx(
                float(solved.loc[pp_id]),
                abs=1e-6,
            )
    assert saw_slack_off_first_row
