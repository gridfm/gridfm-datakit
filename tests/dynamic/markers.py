"""Shared skip markers for the dynamic test suite.

Dynawo is a native install, not a pip dependency, so it is gated separately.
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
