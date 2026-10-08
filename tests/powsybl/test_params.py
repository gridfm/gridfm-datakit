"""Tests for gridfm_datakit.powsybl.params module."""

import pytest

import gridfm_datakit.powsybl as powsybl
from gridfm_datakit.powsybl import get_default_lf_params

pytestmark = pytest.mark.skipif(
    not powsybl.is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


def test_default_lf_params_keep_a_single_slack():
    params = get_default_lf_params()
    assert params.distributed_slack is False


def test_default_lf_params_round_trip_the_slack_bus():
    params = get_default_lf_params()
    assert params.read_slack_bus is True
    assert params.write_slack_bus is True


def test_default_lf_params_put_the_slack_on_a_generator():
    params = get_default_lf_params()
    assert params.provider_parameters["slackBusSelectionMode"] == "LARGEST_GENERATOR"
