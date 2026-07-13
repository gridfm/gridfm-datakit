# TODO: add more tests
from __future__ import annotations

import pytest

from gridfm_datakit.dynamic.dynawo.api import is_dynawo_available

# Full end-to-end pipeline (OPF via Julia + Dynawo). Gate on the Dynawo backend,
# not just pypowsybl: a local Dynawo installation is a separate prerequisite, and
# without it the pipeline raises rather than skipping.
pytestmark = pytest.mark.skipif(
    is_dynawo_available() is False,
    reason="Dynawo backend unavailable (needs pypowsybl + a local Dynawo installation)",
)


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


def test_generate_dynamic(config_ieee14):
    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data

    generate_dynamic_data(config_ieee14)
    assert True
