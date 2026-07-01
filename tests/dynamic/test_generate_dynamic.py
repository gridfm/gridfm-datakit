# TODO: add more tests 
from __future__ import annotations

# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

def test_generate_dynamic(config_ieee14):
    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data
    generate_dynamic_data(config_ieee14)
    assert True
