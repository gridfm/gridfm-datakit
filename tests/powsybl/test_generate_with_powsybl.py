"""Tests for gridfm_datakit.generate using PowSyBl's Open Load Flow."""

import pytest
from pathlib import Path

from gridfm_datakit.powsybl.api import is_powsybl_available

pytestmark = pytest.mark.skipif(
    not is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config_ieee14_m():
    """Load configuration that uses IEEE 14 bus network in .m format."""
    import yaml
    
    configs_dir = Path(__file__).parent/"data"/"configs"
    with open(configs_dir/"config_test_powsybl_IEEE14_m.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="module")
def config_ieee14_mat():
    """Load configuration that uses IEEE 14 bus network in .mat format."""
    import yaml
    
    configs_dir = Path(__file__).parent/"data"/"configs"
    with open(configs_dir/"config_test_powsybl_IEEE14_mat.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="module")
def config_ieee14_raw():
    """Load configuration that uses IEEE 14 bus network in .raw format."""
    import yaml
    
    configs_dir = Path(__file__).parent/"data"/"configs"
    with open(configs_dir/"config_test_powsybl_IEEE14_raw.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="module")
def config_ieee14_xiidm():
    """Load configuration that uses IEEE 14 bus network in .xiidm format."""
    import yaml
    
    configs_dir = Path(__file__).parent/"data"/"configs"
    with open(configs_dir/"config_test_powsybl_IEEE14_xiidm.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="module")
def config_ieee14_cgmes():
    """Load configuration that uses IEEE 14 bus network in cgmes format."""
    import yaml
    
    configs_dir = Path(__file__).parent/"data"/"configs"
    with open(configs_dir/"config_test_powsybl_IEEE14_zip.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

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

        config = config_ieee14_m
        generate_power_flow_data(config)
        assert True

    def test_ieee14_mat(self, config_ieee14_mat):
        """Test handling of MATPOWER format (.mat extension)."""
        from gridfm_datakit import generate_power_flow_data
        
        config = config_ieee14_mat
        generate_power_flow_data(config)
        assert True

    def test_ieee14_raw(self, config_ieee14_raw):
        """Test handling of PSS/E format (.raw extension)."""
        from gridfm_datakit import generate_power_flow_data
        
        config = config_ieee14_raw
        generate_power_flow_data(config)        
        assert True
    
    def test_ieee14_xiidm(self, config_ieee14_xiidm):
        """Test handling of XIIDM format (.xiidm extension)."""
        from gridfm_datakit import generate_power_flow_data
        
        config = config_ieee14_xiidm
        generate_power_flow_data(config)
        assert True
    
    def test_ieee14_cgmes(self, config_ieee14_cgmes):
        """Test handling of CGMES format (.zip extension)."""
        from gridfm_datakit import generate_power_flow_data
        
        config = config_ieee14_cgmes
        generate_power_flow_data(config)
        assert True
