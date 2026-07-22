"""Shared skip markers for the dynamic test suite.

Two prerequisites, gated separately because they fail differently:

- ``needs_powsybl``: pypowsybl, a pip dependency.
- ``needs_dynawo``: pypowsybl *and* a local Dynawo installation, which is native,
  not bundled with pypowsybl, and declared to powsybl through ~/.itools/config.yml.
  Without it ``Simulation.run()`` raises rather than skipping.

Use ``pytestmark = needs_x`` when a whole module needs it, ``@needs_x`` for
individual tests.
"""

import pytest

from gridfm_datakit.dynamic.dynawo.api import (
    is_dynawo_available,
    is_pypowsybl_dynamic_available,
)

needs_powsybl = pytest.mark.skipif(
    is_pypowsybl_dynamic_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[dynamic]",
)

needs_dynawo = pytest.mark.skipif(
    is_dynawo_available() is False,
    reason="Dynawo backend unavailable (needs pypowsybl + a local Dynawo installation)",
)
