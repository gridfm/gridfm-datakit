"""Disconnected elements, which pypowsybl before 1.16 leaves out of the export."""

import pytest

import gridfm_datakit.powsybl as powsybl
from gridfm_datakit.powsybl.convert import update_powsybl
from gridfm_datakit.utils.idx_brch import BR_STATUS, F_BUS, T_BUS
from gridfm_datakit.utils.idx_gen import GEN_BUS, GEN_STATUS

pytestmark = pytest.mark.skipif(
    not powsybl.is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)

_DISCONNECTED_GEN = "B3-G"
_DISCONNECTED_LINE = "L6-11-1"
_DISCONNECTED_LOAD = "B4-L"


@pytest.fixture
def case14_with_disconnections():
    """IEEE 14-bus network with a generator, a line and a load disconnected."""
    import pypowsybl as pp

    pp_net = pp.network.create_ieee14()
    for element_id in (_DISCONNECTED_GEN, _DISCONNECTED_LINE, _DISCONNECTED_LOAD):
        pp_net.disconnect(element_id)
    net = powsybl.from_powsybl(pp_net)
    return net, pp_net, powsybl.build_p2g_maps(net, pp_net)


class TestDisconnectedElements:
    """The maps follow the export, whether it keeps disconnected elements or not."""

    def test_maps_cover_every_row(self, case14_with_disconnections):
        net, _, mapping = case14_with_disconnections
        assert sorted(mapping.gen.values()) == list(range(net.gens.shape[0]))
        assert sorted(mapping.branch.values()) == list(range(net.branches.shape[0]))

    def test_disconnected_rows_are_out_of_service(self, case14_with_disconnections):
        net, _, mapping = case14_with_disconnections
        if _DISCONNECTED_GEN in mapping.gen:
            assert net.gens[mapping.gen[_DISCONNECTED_GEN], GEN_STATUS] == 0
        if _DISCONNECTED_LINE in mapping.branch:
            assert net.branches[mapping.branch[_DISCONNECTED_LINE], BR_STATUS] == 0

    def test_gen_map_follows_gen_buses(self, case14_with_disconnections):
        net, pp_net, mapping = case14_with_disconnections
        bus_id = pp_net.get_generators()["bus_id"]
        for pp_gen_id, row in mapping.gen.items():
            if pp_gen_id != _DISCONNECTED_GEN:
                assert int(mapping.bus[bus_id[pp_gen_id]]) == int(
                    net.gens[row, GEN_BUS],
                )

    def test_branch_map_follows_end_buses(self, case14_with_disconnections):
        net, pp_net, mapping = case14_with_disconnections
        ends = pp_net.get_branches()[["bus1_id", "bus2_id"]]
        for pp_branch_id, row in mapping.branch.items():
            if pp_branch_id != _DISCONNECTED_LINE:
                bus1, bus2 = ends.loc[pp_branch_id]
                assert int(mapping.bus[bus1]) == int(net.branches[row, F_BUS])
                assert int(mapping.bus[bus2]) == int(net.branches[row, T_BUS])

    def test_power_flow_round_trip(self, case14_with_disconnections):
        import pypowsybl as pp

        net, pp_net, mapping = case14_with_disconnections
        update_powsybl(pp_net, net, mapping)
        assert not pp_net.get_generators()["connected"][_DISCONNECTED_GEN]
        assert not pp_net.get_loads()["connected"][_DISCONNECTED_LOAD]
        assert not pp_net.get_lines()["connected1"][_DISCONNECTED_LINE]
        pf_metadata = pp.loadflow.run_ac(pp_net)
        res = powsybl.get_pf_res(pp_net, 0.0, pf_metadata, mapping)
        assert len(res["solution"]["gen"]) == int(net.gens[:, GEN_STATUS].sum())
        assert len(res["solution"]["branch"]) == net.branches.shape[0]
