"""Config-block validation for the Dynawo solver parameters.

These checks run in the parent process, before any worker is spawned. That is the
point of them: a bad key reaching a worker surfaces as a bare KeyError that fails
every chunk with no indication of which setting was wrong.

The rejection tests need no pypowsybl — validation happens before any pypowsybl
object is built. Only the tests that inspect a constructed Parameters object do.
"""

from __future__ import annotations

import pytest

from gridfm_datakit.dynamic.dynawo import get_dynawo_simulation_parameters
from gridfm_datakit.dynamic.dynawo.api import is_pypowsybl_dynamic_available
from gridfm_datakit.utils.param_handler import NestedNamespace

needs_pypowsybl = pytest.mark.skipif(
    is_pypowsybl_dynamic_available() is False,
    reason="pypowsybl.dynamic not installed",
)


def _config(**solver_parameters) -> NestedNamespace:
    return NestedNamespace(
        dynamic={"solver_parameters": solver_parameters},
    )


# ---------------------------------------------------------------------------
# dynamic.solver_parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "present,missing",
    [
        ({"stop_time": 500.0}, "start_time"),
        ({"start_time": 0.0}, "stop_time"),
        ({}, "start_time"),
    ],
)
def test_missing_required_solver_key_raises(present, missing):
    with pytest.raises(ValueError, match=missing):
        get_dynawo_simulation_parameters(_config(**present))


def test_unsupported_solver_key_raises():
    config = _config(start_time=0.0, stop_time=500.0, solver_typo="SIM")
    with pytest.raises(ValueError, match="solver_typo"):
        get_dynawo_simulation_parameters(config)


@needs_pypowsybl
def test_minimal_solver_parameters_are_accepted():
    params = get_dynawo_simulation_parameters(_config(start_time=0.0, stop_time=500.0))
    assert params.start_time == 0.0
    assert params.stop_time == 500.0
