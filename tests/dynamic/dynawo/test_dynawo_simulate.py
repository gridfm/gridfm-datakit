import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# pypowsybl is an optional dependency; guard the import so collection never
# fails when it is absent (the tests below are skipped anyway).
try:
    import pypowsybl as pp
except ImportError:  # pragma: no cover - optional dependency
    pp = None

from markers import needs_dynawo, needs_powsybl

# Most tests only run OPF + AC-PF; Simulation.run() ones carry @needs_dynawo.
pytestmark = needs_powsybl

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


@needs_dynawo
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


def _report(*models) -> str:
    """A ReportNode JSON carrying one instantiation entry per (model, id, state)."""
    return json.dumps(
        {
            "version": "3.0",
            "reportRoot": {
                "messageKey": "",
                "children": [
                    {
                        "messageKey": "dynawo.dynasim.dynawoSimulation",
                        "children": [
                            {
                                "messageKey": "pypowsybl.dynasim.pypowsyblDynamicModels",
                                "children": [
                                    {
                                        "messageKey": "dynawo.dynasim.modelInstantiation",
                                        "values": {
                                            "modelName": {"value": model},
                                            "dynamicId": {"value": dynamic_id},
                                            "state": {"value": state},
                                        },
                                    }
                                    for model, dynamic_id, state in models
                                ],
                            },
                        ],
                    },
                ],
            },
        },
    )


class TestFailedModelInstantiations:
    """A model Dynawo cannot build is skipped, and the run still reports SUCCESS."""

    def test_all_models_instantiated_is_not_a_failure(self):
        from gridfm_datakit.dynamic.dynawo.simulate import _failed_model_instantiations

        report = _report(("GeneratorSynchronous", "_GEN____1_SM", "OK"))
        assert _failed_model_instantiations(report) == []

    def test_a_ko_model_is_reported_with_its_id(self):
        from gridfm_datakit.dynamic.dynawo.simulate import _failed_model_instantiations

        report = _report(
            ("GeneratorSynchronous", "_GEN____1_SM", "OK"),
            ("GeneratorSynchronous", "_LOAD___2_EC", "KO"),
        )
        assert _failed_model_instantiations(report) == [
            "GeneratorSynchronous _LOAD___2_EC",
        ]

    @pytest.mark.parametrize(
        "report",
        ["", "not json", "{}", '{"reportRoot": null}', None, 42],
    )
    def test_an_unreadable_report_is_not_treated_as_a_failure(self, report):
        from gridfm_datakit.dynamic.dynawo.simulate import _failed_model_instantiations

        assert _failed_model_instantiations(report) == []


@needs_dynawo
def test_a_model_that_cannot_be_instantiated_raises(
    pp_net_ieee14,
    event_mapping_ieee14,
    variable_mapping_ieee14,
    param_ieee14,
):
    """Dynawo returns SUCCESS, the pipeline must not."""
    from gridfm_datakit.dynamic.dynawo import DynawoMappings
    from gridfm_datakit.dynamic.dynawo.simulate import run_dynawo_simulation

    # A load equipped with a generator model, among four sound machines: the rest
    # of the system is intact, so the run still converges.
    model_mapping = pp.dynamic.ModelMapping()
    model_mapping.add_dynamic_model(
        category_name="SynchronousGenerator",
        df=pd.DataFrame.from_records(
            index="static_id",
            columns=["static_id", "parameter_set_id", "model_name"],
            data=[
                (
                    "_LOAD___2_EC",  # not a generator
                    "Generator1",
                    "GeneratorSynchronousFourWindingsProportionalRegulations",
                ),
                (
                    "_GEN____2_SM",
                    "Generator2",
                    "GeneratorSynchronousFourWindingsProportionalRegulations",
                ),
                (
                    "_GEN____3_SM",
                    "Generator3",
                    "GeneratorSynchronousFourWindingsProportionalRegulations",
                ),
                (
                    "_GEN____6_SM",
                    "Generator6",
                    "GeneratorSynchronousThreeWindingsProportionalRegulations",
                ),
                (
                    "_GEN____8_SM",
                    "Generator8",
                    "GeneratorSynchronousThreeWindingsProportionalRegulations",
                ),
            ],
        ),
    )

    with pytest.raises(RuntimeError, match="failed to instantiate"):
        run_dynawo_simulation(
            pp_net=pp_net_ieee14,
            dynawo_mapping=DynawoMappings(
                dynamic_model_mapping=model_mapping,
                event_mapping=event_mapping_ieee14,
                variable_mapping=variable_mapping_ieee14,
            ),
            parameters=param_ieee14,
        )


@needs_dynawo
def test_a_diverged_simulation_raises(
    pp_net_ieee14,
    event_mapping_ieee14,
    variable_mapping_ieee14,
    param_ieee14,
):
    """Dynawo returns a FAILURE result instead of raising."""
    from gridfm_datakit.dynamic.dynawo import DynawoMappings
    from gridfm_datakit.dynamic.dynawo.simulate import run_dynawo_simulation

    # No machine carries a dynamic model, so initialisation has nothing to solve.
    model_mapping = pp.dynamic.ModelMapping()
    model_mapping.add_dynamic_model(
        category_name="SynchronousGenerator",
        df=pd.DataFrame.from_records(
            index="static_id",
            columns=["static_id", "parameter_set_id", "model_name"],
            data=[
                (
                    "_LOAD___2_EC",
                    "Generator1",
                    "GeneratorSynchronousFourWindingsProportionalRegulations",
                ),
            ],
        ),
    )

    with pytest.raises(RuntimeError, match="Dynawo simulation failed"):
        run_dynawo_simulation(
            pp_net=pp_net_ieee14,
            dynawo_mapping=DynawoMappings(
                dynamic_model_mapping=model_mapping,
                event_mapping=event_mapping_ieee14,
                variable_mapping=variable_mapping_ieee14,
            ),
            parameters=param_ieee14,
        )


def test_failed_simulation_raises(unit_test_param):
    """Test that a failed Dynawo simulation raises correctly."""

    import pandas as pd
    from gridfm_datakit.dynamic.dynawo.simulate import run_dynawo_simulation
    from gridfm_datakit.dynamic.dynawo import (
        DynawoMappings,
        _map_dynamic_models_dynawo,
    )

    pp_net = pp.network.create_ieee9()
    df_static_element_models = pd.DataFrame.from_records(
        columns=["category_name", "static_id", "parameter_set_id", "model_name"],
        data=[
            ("Wecc", "B1-G", "WECC_WTG4A", "WTG4AWeccCurrentSource"),
            (
                "Wecc",
                "B2-G",
                "WECC_WTG4A",
                "WTG4AWeccCurrentSource",
            ),
        ],
    )

    df_automation_systems = pd.DataFrame(
        columns=[
            "category_name",
            "dynamic_model_id",
            "parameter_set_id",
            "params",
            "model_name",
        ],
    )
    with pytest.raises(RuntimeError):
        run_dynawo_simulation(
            pp_net=pp_net,
            dynawo_mapping=DynawoMappings(
                dynamic_model_mapping=_map_dynamic_models_dynawo(
                    [df_static_element_models, df_automation_systems],
                ),
                event_mapping=None,
                variable_mapping=None,
            ),
            parameters=unit_test_param,
            drop_duplicate_timestep=True,
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
