# TODO: add more tests
from __future__ import annotations

import pytest

from gridfm_datakit.powsybl.api import is_powsybl_available

# Full end-to-end pipeline (OPF via Julia + Dynawo), so it needs pypowsybl.
pytestmark = pytest.mark.skipif(
    is_powsybl_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


def test_generate_dynamic(config_ieee14):
    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data

    generate_dynamic_data(config_ieee14)
    assert True
