"""Tests for gridfm_datakit.generate using PowSyBl's Open Load Flow."""

import pytest
from pathlib import Path

from gridfm_datakit.powsybl.api import is_powsybl_available

pytestmark = pytest.mark.skipif(
    not is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)

_GRIDS_DIR = Path(__file__).parent / "grids"
_DATA_DIR = "./tests/powsybl/data/generated_test_data"

_BASE_CONFIG = {
    "network": {
        "source": "file",
        "reader": "powsybl",
        "network_dir": "scripts/grids",
    },
    "load": {
        "generator": "agg_load_profile",
        "agg_profile": "default",
        "scenarios": 5,
        "sigma": 0.2,
        "change_reactive_power": True,
        "global_range": 0.4,
        "max_scaling_factor": 4.0,
        "step_size": 0.05,
        "start_scaling_factor": 0.8,
    },
    "topology_perturbation": {
        "type": "none",
        "k": 2,
        "n_topology_variants": 2,
        "elements": ["branch", "gen"],
    },
    "generation_perturbation": {
        "type": "cost_permutation",
        "sigma": 1.0,
    },
    "admittance_perturbation": {
        "type": "random_perturbation",
        "sigma": 0.2,
    },
    "settings": {
        "num_processes": 1,
        "data_dir": _DATA_DIR,
        "large_chunk_size": 5,
        "overwrite": True,
        "mode": "pf",
        "include_dc_res": True,
        "enable_solver_logs": False,
        "pf_fast": False,
        "dcpf_fast": False,
        "max_iter": 200,
        "pf_solver": "powsybl",
        "seed": 49455,
    },
}


def _make_config(name: str, grid_file: str) -> dict:
    import copy
    config = copy.deepcopy(_BASE_CONFIG)
    config["network"]["name"] = name
    config["network"]["file"] = str(_GRIDS_DIR / grid_file)
    return config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config_ieee14_m():
    return _make_config("IEEE14_m", "ieee14.m")


@pytest.fixture(scope="module")
def config_ieee14_mat():
    return _make_config("IEEE14_mat", "ieee14.mat")


@pytest.fixture(scope="module")
def config_ieee14_raw():
    return _make_config("IEEE14_raw", "ieee14.raw")


@pytest.fixture(scope="module")
def config_ieee14_xiidm():
    return _make_config("IEEE14_xiidm", "ieee14.xiidm")


@pytest.fixture(scope="module")
def config_ieee14_cgmes():
    return _make_config("IEEE14_cgmes", "ieee14.zip")


# ---------------------------------------------------------------------------
# 1. Format tests
# ---------------------------------------------------------------------------

class TestFormats:
    """
    Tests the ability to handle MATPOWER (.m and .mat extensions), PSS/E, XIIDM and CGMES formats.
    """
    def test_ieee14_m(self, config_ieee14_m):
        "Test handling of MATPOWER format (.m extension)"
        from gridfm_datakit import generate_power_flow_data

        generate_power_flow_data(config_ieee14_m)
        assert True

    def test_ieee14_mat(self, config_ieee14_mat):
        """Test handling of MATPOWER format (.mat extension)."""
        from gridfm_datakit import generate_power_flow_data

        generate_power_flow_data(config_ieee14_mat)
        assert True

    def test_ieee14_raw(self, config_ieee14_raw):
        """Test handling of PSS/E format (.raw extension)."""
        from gridfm_datakit import generate_power_flow_data

        generate_power_flow_data(config_ieee14_raw)
        assert True

    def test_ieee14_xiidm(self, config_ieee14_xiidm):
        """Test handling of XIIDM format (.xiidm extension)."""
        from gridfm_datakit import generate_power_flow_data

        generate_power_flow_data(config_ieee14_xiidm)
        assert True

    def test_ieee14_cgmes(self, config_ieee14_cgmes):
        """Test handling of CGMES format (.zip extension)."""
        from gridfm_datakit import generate_power_flow_data

        generate_power_flow_data(config_ieee14_cgmes)
        assert True
