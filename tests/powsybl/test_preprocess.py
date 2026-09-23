"""Tests for gridfm_datakit.powsybl.preprocess_pf_res module."""

import time

import pytest

import gridfm_datakit.powsybl as powsybl
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.powsybl.preprocess import _is_power_flow_computed

pytestmark = pytest.mark.skipif(
    not powsybl.is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def ieee14_acpf_res():
    """ACPF on IEEE14 with default configuration."""
    gfm_net = load_net_from_pglib("case14_ieee")
    conv = powsybl.to_powsybl(gfm_net)
    pp_net = conv.pp_net

    start_time = time.perf_counter()
    pf_metadata = powsybl.pypowsybl.loadflow.run_ac(
        pp_net,
        powsybl.get_default_lf_params(),
    )
    end_time = time.perf_counter()
    solve_time = end_time - start_time

    return pp_net, solve_time, pf_metadata, conv.mapping_p2g


@pytest.fixture(scope="function")
def ieee14_acpf_non_convergent_res():
    """ACPF on a IEEE14 network that should not converge because of an excessive load impossible to balance."""
    gfm_net = load_net_from_pglib("case14_ieee")
    conv = powsybl.to_powsybl(gfm_net)
    pp_net = conv.pp_net

    pp_net.update_loads(id="LOAD-2", p0=4000)
    start_time = time.perf_counter()
    pf_metadata = powsybl.pypowsybl.loadflow.run_ac(
        pp_net,
        powsybl.get_default_lf_params(),
    )
    end_time = time.perf_counter()
    solve_time = end_time - start_time

    return pp_net, solve_time, pf_metadata, conv.mapping_p2g


# ---------------------------------------------------------------------------
# 1. Structural tests
# ---------------------------------------------------------------------------


class TestPreprocessPPPFRes:
    """Structural correctness of preprocess_pp_pf_res."""

    def test_bus_coverage(self, ieee14_acpf_res):
        """All buses' results are converted through preprocessing."""

        pp_net, solve_time, pf_metadata, mapping_p2g = ieee14_acpf_res
        res = powsybl.get_pf_res(pp_net, solve_time, pf_metadata, mapping_p2g)
        pp_net.per_unit = True
        bus_res = res["solution"]["bus"]
        pp_bus_res = pp_net.get_buses()
        for idx_pp, idx_gfm in mapping_p2g.bus.items():
            assert (
                bus_res[str(int(idx_gfm + 1))]["vm"] == pp_bus_res.loc[idx_pp]["v_mag"]
            )
            assert (
                bus_res[str(int(idx_gfm + 1))]["va"]
                == pp_bus_res.loc[idx_pp]["v_angle"]
            )

    def test_branch_converage(self, ieee14_acpf_res):
        """All branches's results are converted through preprocessing."""

        pp_net, solve_time, pf_metadata, mapping_p2g = ieee14_acpf_res
        res = powsybl.get_pf_res(pp_net, solve_time, pf_metadata, mapping_p2g)
        pp_net.per_unit = True

        branch_res = res["solution"]["branch"]
        pp_branch_res = pp_net.get_branches()

        for idx_pp, idx_gfm in mapping_p2g.branch.items():
            assert (
                branch_res[str(int(idx_gfm + 1))]["pf"]
                == pp_branch_res.loc[idx_pp]["p1"]
            )
            assert (
                branch_res[str(int(idx_gfm + 1))]["qf"]
                == pp_branch_res.loc[idx_pp]["q1"]
            )
            assert (
                branch_res[str(int(idx_gfm + 1))]["pt"]
                == pp_branch_res.loc[idx_pp]["p2"]
            )
            assert (
                branch_res[str(int(idx_gfm + 1))]["qt"]
                == pp_branch_res.loc[idx_pp]["q2"]
            )

    def test_gen_converage(self, ieee14_acpf_res):
        """All generators' results are converted through preprocessing."""

        pp_net, solve_time, pf_metadata, mapping_p2g = ieee14_acpf_res
        res = powsybl.get_pf_res(pp_net, solve_time, pf_metadata, mapping_p2g)
        pp_net.per_unit = True

        slack_bus = pf_metadata[0].slack_bus_results[0].id
        slack_res = pf_metadata[0].slack_bus_results[0].active_power_mismatch

        gen_res = res["solution"]["gen"]
        pp_gen_res = pp_net.get_generators()
        slack_gen_id = pp_gen_res[pp_gen_res["bus_id"] == slack_bus].index[0]

        for idx_pp, idx_gfm in mapping_p2g.gen.items():
            if idx_pp == slack_gen_id:
                assert (
                    gen_res[str(int(idx_gfm + 1))]["pg"]
                    == -pp_gen_res.loc[idx_pp]["p"]
                    + slack_res / pp_net.nominal_apparent_power
                )
                assert (
                    gen_res[str(int(idx_gfm + 1))]["qg"] == -pp_gen_res.loc[idx_pp]["q"]
                )
            else:
                assert (
                    gen_res[str(int(idx_gfm + 1))]["pg"] == -pp_gen_res.loc[idx_pp]["p"]
                )
                assert (
                    gen_res[str(int(idx_gfm + 1))]["qg"] == -pp_gen_res.loc[idx_pp]["q"]
                )

    def test_base_power(self, ieee14_acpf_res):
        """Base MVA should be included in preprocessed results."""

        pp_net, solve_time, pf_metadata, mapping_p2g = ieee14_acpf_res
        res = powsybl.get_pf_res(pp_net, solve_time, pf_metadata, mapping_p2g)
        assert res["solution"]["baseMVA"] == pp_net.nominal_apparent_power

    def test_solve_time(self, ieee14_acpf_res):
        """Solve time should be included in preprocessed results."""

        pp_net, solve_time, pf_metadata, mapping_p2g = ieee14_acpf_res
        res = powsybl.get_pf_res(pp_net, solve_time, pf_metadata, mapping_p2g)
        assert res["solve_time"] == solve_time

    def test_pf_status(self, ieee14_acpf_res):
        """Power flow status should be included in preprocessed results."""

        pp_net, solve_time, pf_metadata, mapping_p2g = ieee14_acpf_res
        res = powsybl.get_pf_res(pp_net, solve_time, pf_metadata, mapping_p2g)
        assert res["solution"]["pf"] == _is_power_flow_computed(
            pf_metadata[0].status_text,
        )


# ---------------------------------------------------------------------------
# 2. Non convergence
# ---------------------------------------------------------------------------


class TestNonConvergence:
    """In case of non convergence, the preprocess function should return a ValueError."""

    def test_non_convergence(self, ieee14_acpf_non_convergent_res):
        """Non convergent power flow result should raise a ValueError."""

        pp_net, solve_time, pf_metadata, mapping_p2g = ieee14_acpf_non_convergent_res
        pf_status = pf_metadata[0].status_text
        with pytest.raises(
            ValueError,
            match=f"Power flow computation failed. The returned power flow status:{pf_status}",
        ):
            powsybl.get_pf_res(pp_net, solve_time, pf_metadata, mapping_p2g)
