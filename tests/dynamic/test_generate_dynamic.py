# TODO: add more tests
from __future__ import annotations

from pathlib import Path

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

    file_paths = generate_dynamic_data(config_ieee14)
    assert Path(file_paths["dynamic_results"]).exists()
    assert Path(file_paths["metadata"]).exists()


def test_generated_data_passes_static_validation(config_ieee14):
    """The dynamic pipeline's PF snapshot must satisfy the same physics checks
    (Y-bus consistency, Kirchhoff balance, generator limits, ...) as the static
    pipeline's output."""
    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data
    from gridfm_datakit.validation import validate_dynamic_data

    file_paths = generate_dynamic_data(config_ieee14)
    assert validate_dynamic_data(file_paths, mode="pf", sn_mva=100.0)


def test_curves_carry_a_time_axis_in_seconds(config_ieee14):
    """Curves are unusable as labels without knowing which instant each column is."""
    import numpy as np
    import zarr

    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data

    file_paths = generate_dynamic_data(config_ieee14)
    store = zarr.open(file_paths["dynamic_results"], mode="r")
    curves = np.asarray(store["curves"])
    time = np.asarray(store["time"])

    # one time value per curve column, per scenario
    assert time.shape == (curves.shape[0], curves.shape[2])
    # and it is the config's simulation window, in seconds
    sp = config_ieee14.dynamic.solver_parameters
    assert np.nanmin(time) == sp.start_time
    assert np.nanmax(time) == sp.stop_time
