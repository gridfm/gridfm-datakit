"""Config-block validation for the Dynawo solver and load flow parameters."""

from __future__ import annotations

import pytest
from markers import needs_powsybl

from gridfm_datakit.dynamic.dynawo import (
    get_dynawo_loadflow_parameters,
    get_dynawo_simulation_parameters,
)
from gridfm_datakit.dynamic.dynawo.utils import LOADFLOW_PARAMETERS_DEFAULTS
from gridfm_datakit.utils.param_handler import NestedNamespace


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


@needs_powsybl
def test_minimal_solver_parameters_are_accepted():
    params = get_dynawo_simulation_parameters(_config(start_time=0.0, stop_time=500.0))
    assert params.start_time == 0.0
    assert params.stop_time == 500.0


# ---------------------------------------------------------------------------
# dynamic.loadflow_parameters
# ---------------------------------------------------------------------------


def test_unsupported_loadflow_key_raises():
    config = NestedNamespace(
        dynamic={"loadflow_parameters": {"distributed_slak": True}},
    )
    with pytest.raises(ValueError, match="distributed_slak"):
        get_dynawo_loadflow_parameters(config)


@needs_powsybl
def test_loadflow_defaults_put_the_slack_on_a_generator():
    params = get_dynawo_loadflow_parameters(NestedNamespace())
    assert params.distributed_slack is False
    assert params.provider_parameters["slackBusSelectionMode"] == "LARGEST_GENERATOR"


@needs_powsybl
def test_absent_block_matches_explicit_defaults():
    implicit = get_dynawo_loadflow_parameters(NestedNamespace())
    explicit = get_dynawo_loadflow_parameters(
        NestedNamespace(dynamic={"loadflow_parameters": LOADFLOW_PARAMETERS_DEFAULTS}),
    )
    assert implicit.distributed_slack == explicit.distributed_slack
    assert implicit.provider_parameters == explicit.provider_parameters


@needs_powsybl
def test_loadflow_overrides_are_applied():
    config = NestedNamespace(
        dynamic={
            "loadflow_parameters": {
                "distributed_slack": True,
                "provider_parameters": {"slackBusSelectionMode": "MOST_MESHED"},
            },
        },
    )
    params = get_dynawo_loadflow_parameters(config)
    assert params.distributed_slack is True
    assert params.provider_parameters["slackBusSelectionMode"] == "MOST_MESHED"
    assert params.read_slack_bus is True  # not overridden, keeps its default
