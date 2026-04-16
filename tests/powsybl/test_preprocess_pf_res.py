"""Tests for gridfm_datakit.powsybl.preprocess_pf_res module."""

import pytest
from pathlib import Path

from gridfm_datakit.powsybl.api import is_powsybl_available
from gridfm_datakit.powsybl import load_net

pytestmark = pytest.mark.skipif(
    not is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def ieee14_acpf_res():
    """ACPF on IEEE14 with default configuration."""
    import pypowsybl as pp
    import time
    from gridfm_datakit.powsybl.utils.lf_parameters import get_default_lf_parameters
    from gridfm_datakit.powsybl.mapping import to_powsybl_with_mapping
    
    grids_dir = Path(__file__).parent/"data"/"grids"
    loaded_net = load_net(str(grids_dir/"ieee14.m"))

    pp_net = loaded_net.pp_net
    gfm_net = loaded_net.gfm_net

    start_time = time.perf_counter()
    pf_metadata = pp.loadflow.run_ac(pp_net, get_default_lf_parameters())
    end_time = time.perf_counter()
    solve_time = end_time - start_time

    _, map_bus_p2g, map_branch_p2g, map_gen_p2g = to_powsybl_with_mapping(gfm_net)
    
    return pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g

@pytest.fixture(scope="function")
def ieee14_acpf_non_convergent_res():
    """ACPF on a IEEE14 network that should not converge because of an excessive load impossible to balance."""
    import pypowsybl as pp
    import time
    from gridfm_datakit.powsybl.utils.lf_parameters import get_default_lf_parameters
    from gridfm_datakit.powsybl.mapping import to_powsybl_with_mapping
    
    grids_dir = Path(__file__).parent/"data"/"grids"
    loaded_net = load_net(str(grids_dir/"ieee14.m"))

    pp_net = loaded_net.pp_net
    gfm_net = loaded_net.gfm_net

    pp_net.update_loads(id='LOAD-2', p0=4000)
    start_time = time.perf_counter()
    pf_metadata = pp.loadflow.run_ac(pp_net, get_default_lf_parameters())
    end_time = time.perf_counter()
    solve_time = end_time - start_time

    _, map_bus_p2g, map_branch_p2g, map_gen_p2g = to_powsybl_with_mapping(gfm_net)
    
    return pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g


# ---------------------------------------------------------------------------
# 1. Structural tests
# ---------------------------------------------------------------------------

class TestPreprocessPPPFRes:
    """Structural correctness of preprocess_pp_pf_res."""

    def test_bus_coverage(self, ieee14_acpf_res):
        """All buses' results are converted through preprocessing."""
        from gridfm_datakit.powsybl.preprocess_pf_res import preprocess_pp_pf_res

        pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g = ieee14_acpf_res
        res = preprocess_pp_pf_res(pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g)
        pp_net.per_unit = True
        bus_res = res['solution']['bus']
        pp_bus_res = pp_net.get_buses()
        for idx_pp, idx_gfm in map_bus_p2g.items():
            assert bus_res[str(int(idx_gfm + 1))]['vm'] == pp_bus_res.loc[idx_pp]['v_mag']
            assert bus_res[str(int(idx_gfm + 1))]['va'] == pp_bus_res.loc[idx_pp]['v_angle']

    def test_branch_converage(self, ieee14_acpf_res):
        """All branches's results are converted through preprocessing."""
        from gridfm_datakit.powsybl.preprocess_pf_res import preprocess_pp_pf_res

        pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g = ieee14_acpf_res
        res = preprocess_pp_pf_res(pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g)
        pp_net.per_unit = True

        branch_res = res['solution']['branch']
        pp_branch_res = pp_net.get_branches()

        for idx_pp, idx_gfm in map_branch_p2g.items():
            assert branch_res[str(int(idx_gfm + 1))]['pf'] == pp_branch_res.loc[idx_pp]['p1']
            assert branch_res[str(int(idx_gfm + 1))]['qf'] == pp_branch_res.loc[idx_pp]['q1']
            assert branch_res[str(int(idx_gfm + 1))]['pt'] == pp_branch_res.loc[idx_pp]['p2']
            assert branch_res[str(int(idx_gfm + 1))]['qt'] == pp_branch_res.loc[idx_pp]['q2']
    
    def test_gen_converage(self, ieee14_acpf_res):
        """All generators' results are converted through preprocessing."""
        from gridfm_datakit.powsybl.preprocess_pf_res import preprocess_pp_pf_res

        pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g = ieee14_acpf_res
        res = preprocess_pp_pf_res(pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g)
        pp_net.per_unit = True

        slack_bus = pf_metadata[0].slack_bus_results[0].id
        slack_res = pf_metadata[0].slack_bus_results[0].active_power_mismatch
        
        gen_res = res['solution']['gen']
        pp_gen_res = pp_net.get_generators()
        slack_gen_id = pp_gen_res[pp_gen_res['bus_id'] == slack_bus].index[0]

        for idx_pp, idx_gfm in map_gen_p2g.items():
            if idx_pp == slack_gen_id:
                assert gen_res[str(int(idx_gfm + 1))]['pg'] == -pp_gen_res.loc[idx_pp]['p'] + slack_res/pp_net.nominal_apparent_power
                assert gen_res[str(int(idx_gfm + 1))]['qg'] == -pp_gen_res.loc[idx_pp]['q']
            else:
                assert gen_res[str(int(idx_gfm + 1))]['pg'] == -pp_gen_res.loc[idx_pp]['p']
                assert gen_res[str(int(idx_gfm + 1))]['qg'] == -pp_gen_res.loc[idx_pp]['q']

    def test_base_power(self, ieee14_acpf_res):
        """Base MVA should be included in preprocessed results."""
        from gridfm_datakit.powsybl.preprocess_pf_res import preprocess_pp_pf_res

        pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g = ieee14_acpf_res
        res = preprocess_pp_pf_res(pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g)
        assert res['solution']['baseMVA'] == pp_net.nominal_apparent_power

    def test_solve_time(self, ieee14_acpf_res):
        """Solve time should be included in preprocessed results."""
        from gridfm_datakit.powsybl.preprocess_pf_res import preprocess_pp_pf_res

        pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g = ieee14_acpf_res
        res = preprocess_pp_pf_res(pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g)
        assert res['solve_time'] == solve_time
    
    def test_pf_status(self, ieee14_acpf_res):
        """Power flow status should be included in preprocessed results."""
        from gridfm_datakit.powsybl.preprocess_pf_res import preprocess_pp_pf_res, _is_power_flow_computed

        pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g = ieee14_acpf_res
        res = preprocess_pp_pf_res(pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g)
        assert res['solution']['pf'] == _is_power_flow_computed(pf_metadata[0].status_text)


# ---------------------------------------------------------------------------
# 2. Non convergence
# ---------------------------------------------------------------------------

class TestNonConvergence:
    """In cas of no converge, the preprocess function should return a ValueError."""

    def test_non_convergence(self, ieee14_acpf_non_convergent_res):
        """Non convergent power flow result should raise a ValueError."""
        from gridfm_datakit.powsybl.preprocess_pf_res import preprocess_pp_pf_res

        pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g = ieee14_acpf_non_convergent_res
        pf_status = pf_metadata[0].status_text
        with pytest.raises(ValueError, match=f'Power flow computation failed. The returned power flow status:{pf_status}'):
            preprocess_pp_pf_res(pp_net, solve_time, pf_metadata, map_bus_p2g, map_branch_p2g, map_gen_p2g)
