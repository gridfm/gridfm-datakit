from pathlib import Path

import numpy as np
import pytest

# pypowsybl is an optional dependency; guard the import so collection never
# fails when it is absent (the tests below are skipped anyway).
try:
    import pypowsybl as pp
except ImportError:  # pragma: no cover - optional dependency
    pp = None

from gridfm_datakit.dynamic.dynawo.api import is_dynawo_available
from gridfm_datakit.powsybl.api import is_powsybl_available

# Most tests here only run OPF + AC-PF (no Dynawo), so pypowsybl alone is enough.
pytestmark = pytest.mark.skipif(
    is_powsybl_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)

# Tests that actually call Simulation.run() additionally need a local Dynawo install.
requires_dynawo = pytest.mark.skipif(
    is_dynawo_available() is False,
    reason="Dynawo backend unavailable (needs pypowsybl + a local Dynawo installation)",
)

# IEEE14 network whose buses are NOT exported in sorted-ID order — the fixture
# that makes the format-independence test below meaningful.
PATH_NETWORK_IEEE14 = str(
    Path(__file__).parent
    / "benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.iidm",
)


@pytest.fixture(scope="session")
def julia():
    from gridfm_datakit.generate import init_julia

    return init_julia(200)


@pytest.fixture(scope="function")
def pp_net_ieee9():
    return pp.network.create_ieee9()


def test_compute_balanced_static_state_dynawo_output_formats(pp_net_ieee9, julia):
    from gridfm_datakit.dynamic.dynawo.simulate import (
        compute_balanced_static_state_dynawo,
    )
    from gridfm_datakit.powsybl import from_powsybl, build_p2g_maps

    gfm_net = from_powsybl(pp_net_ieee9)
    p2g_maps = build_p2g_maps(gfm_net, pp_net_ieee9)
    pp_net, pf_data = compute_balanced_static_state_dynawo(
        pp_net_ieee9,
        from_powsybl(pp_net_ieee9),
        julia,
        p2g_maps,
    )
    assert isinstance(pp_net, pp.network.Network) and isinstance(pf_data, dict)


def test_pg_bus_assignment_format_independent(julia):
    """Bus values must be assigned by ID (via p2g_maps), not by row position.

    Uses a network whose pypowsybl bus export order is NOT the sorted-ID order,
    so a naive "bus number N -> array row N" assignment would place voltages and
    generator injections on the wrong buses. Guards the balanced initial state
    fed to Dynawo against silently-scrambled buses.
    """
    from gridfm_datakit.dynamic.dynawo.simulate import (
        compute_balanced_static_state_dynawo,
    )
    from gridfm_datakit.powsybl import load_net
    from gridfm_datakit.utils.column_names import BUS_COLUMNS

    net = load_net(PATH_NETWORK_IEEE14)
    bus_map = net.mapping_p2g.bus  # pp bus id -> gfm bus index

    # Precondition: the fixture actually exercises the non-sorted case, i.e. the
    # gfm index is not the physical bus number (else the test proves nothing).
    export_ids = list(net.pp_net.get_buses().index)
    assert export_ids != sorted(export_ids), "fixture must be non-sorted"
    gfm_idx_to_id = {v: k for k, v in bus_map.items()}

    pp_net, pf_data = compute_balanced_static_state_dynawo(
        net.pp_net,
        net.gfm_net,
        julia,
        net.mapping_p2g,
        0,
    )
    bus = np.asarray(pf_data["bus"])
    c_busidx = BUS_COLUMNS.index("bus")  # 0-based gfm bus index
    c_vm = BUS_COLUMNS.index("Vm")
    c_pg = BUS_COLUMNS.index("Pg")

    # Solved per-unit voltages read from pypowsybl, keyed by physical bus id.
    was = pp_net.per_unit
    pp_net.per_unit = True
    solved_vm = {bid: r.v_mag for bid, r in pp_net.get_buses()[["v_mag"]].iterrows()}
    pp_net.per_unit = was

    # (1) Voltage at each output row must equal the solved voltage of the pp bus
    # whose gfm index labels that row (matched through the map, i.e. by ID).
    for row in bus:
        gfm_idx = int(row[c_busidx])
        expected_vm = solved_vm[gfm_idx_to_id[gfm_idx]]
        assert abs(row[c_vm] - expected_vm) < 1e-6, (
            f"Vm mismatch at gfm bus {gfm_idx}: {row[c_vm]} vs solved {expected_vm}"
        )

    # (2) Generator active injection must appear on generator buses only. The
    # generator buses are resolved from pypowsybl (physical id -> gfm index via
    # the map), independently of the gfm network's internal GEN_BUS array.
    gens = net.pp_net.get_generators()
    gen_gfm_indices = {bus_map[b] for b in gens["bus_id"] if b in bus_map}
    # the mapping is non-trivial (physical bus number != gfm index) — otherwise
    # this would not discriminate a row-order bug
    assert gen_gfm_indices != set(range(len(gen_gfm_indices)))
    pg_by_idx = {int(row[c_busidx]): row[c_pg] for row in bus}
    for gfm_idx, pg in pg_by_idx.items():
        if gfm_idx in gen_gfm_indices:
            assert pg != 0.0, f"expected non-zero Pg at generator bus {gfm_idx}"
        else:
            assert pg == 0.0, f"expected zero Pg at non-generator bus {gfm_idx}"


# Test run_dynawo_simulation using ieee14 benchmark


@requires_dynawo
def test_benchmark_ieee14_run_dynawo_simulation(
    pp_net_ieee14,
    model_mapping_ieee14,
    event_mapping_ieee14,
    variable_mapping_ieee14,
    param_ieee14,
    df_ref_curves_ieee14,
):
    from gridfm_datakit.dynamic.dynawo.simulate import run_dynawo_simulation
    from gridfm_datakit.dynamic.dynawo import DynawoMappings

    drop_duplicate_timestep = True
    dynamic_results = run_dynawo_simulation(
        pp_net=pp_net_ieee14,
        dynawo_mapping=DynawoMappings(
            dynamic_model_mapping=model_mapping_ieee14,
            event_mapping=event_mapping_ieee14,
            variable_mapping=variable_mapping_ieee14,
        ),
        parameters=param_ieee14,
        drop_duplicate_timestep=drop_duplicate_timestep,
    )
    assert _validate_res_against_ref(
        dynamic_results,
        df_ref_curves_ieee14,
        drop_duplicate_timestep,
    )


def _validate_res_against_ref(res, df_ref, drop_duplicate_timestep):
    df_res = res.dynamic_results.reset_index(drop=True).rename(
        columns={
            "_GEN____1_SM_generator_efdPu_value": "GEN____1_SM_generator_efdPu_value",
            "_GEN____1_SM_voltageRegulator_EfdMaxPu": "GEN____1_SM_voltageRegulator_EfdMaxPu",
            "_GEN____3_SM_generator_UPu": "GEN____3_SM_generator_UPu",
            "_GEN____3_SM_generator_efdPu_value": "GEN____3_SM_generator_efdPu_value",
            "_GEN____3_SM_voltageRegulator_EfdMaxPu": "GEN____3_SM_voltageRegulator_EfdMaxPu",
        },
    )
    if drop_duplicate_timestep:
        df_ref = df_ref.set_index("time")
        df_ref = df_ref[~df_ref.index.duplicated(keep="last")]
    df_ref = df_ref.reset_index(drop=True)
    df_ref = df_ref[df_res.columns]
    return df_res.equals(df_ref)
