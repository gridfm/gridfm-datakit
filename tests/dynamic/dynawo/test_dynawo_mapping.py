import pandas as pd

# pypowsybl is an optional dependency; guard the import so collection never
# fails when it is absent (the tests below are skipped anyway).
try:
    import pypowsybl as pp
except ImportError:  # pragma: no cover - optional dependency
    pp = None

from markers import needs_dynawo

from gridfm_datakit.dynamic.dynawo.simulate import _failed_model_instantiations

pytestmark = needs_dynawo

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _validate_output_curves_against_ref(res, df_ref_curves_ieee14):
    df_res = (
        res.curves()
        .reset_index(drop=True)
        .rename(
            columns={
                "_GEN____1_SM_generator_efdPu_value": "GEN____1_SM_generator_efdPu_value",
                "_GEN____1_SM_voltageRegulator_EfdMaxPu": "GEN____1_SM_voltageRegulator_EfdMaxPu",
                "_GEN____3_SM_generator_UPu": "GEN____3_SM_generator_UPu",
                "_GEN____3_SM_generator_efdPu_value": "GEN____3_SM_generator_efdPu_value",
                "_GEN____3_SM_voltageRegulator_EfdMaxPu": "GEN____3_SM_voltageRegulator_EfdMaxPu",
            },
        )
    )
    df_ref = df_ref_curves_ieee14.reset_index(drop=True)
    df_ref = df_ref[df_res.columns]
    return df_res.equals(df_ref)


# ---------------------------------------------------------------------------
# Tests for the mappers
# ---------------------------------------------------------------------------
# pypowsybl.dynamic.xxxMapping objects don't support native comparison between two instances,
# To test the relevance of the mappers, we run full dynawo simulations on a benchmark test case.
# The tests comprise:
#  - a baseline test checking that the simulation on baseline inputs
#    produces results identical to benchmark.
#  - individual tests on the mappers: for each tested mapper (model mapper, event mapper, variable mapper),
#    the mapping produced by the mapper replaces the corresponding baseline input mapping,
#    then a full dynamic simulation is run and the results are compared against the benchmark.
#  - testings of the mappings for the supported static element dynamic models, automation systems, events and variables.


# Testing the baseline, which takes the inputs given directly by the conftest file
def test_baseline(
    pp_net_ieee14,
    model_mapping_ieee14,
    event_mapping_ieee14,
    variable_mapping_ieee14,
    param_ieee14,
    df_ref_curves_ieee14,
):
    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()
    res = sim.run(
        network=pp_net_ieee14,
        model_mapping=model_mapping_ieee14,
        event_mapping=event_mapping_ieee14,
        timeseries_mapping=variable_mapping_ieee14,
        parameters=param_ieee14,
        report_node=report_node,
    )
    assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)


class TestDynamicModelMapping:
    """Tests for the dynamic model mapper."""

    def test_dynamic_model_mapping_on_benchmark(
        self,
        pp_net_ieee14,
        event_mapping_ieee14,
        variable_mapping_ieee14,
        param_ieee14,
        df_ref_curves_ieee14,
        benchmark_dataset,
    ):
        """Test the model mapping on the benchmark.

        It comprises the mapping of load, synchronous generators
        and an undervoltage automation system.
        """
        from gridfm_datakit.dynamic.dynawo import _map_dynamic_models_dynawo

        df_static_element_models = benchmark_dataset["df_static_element_dynamic_models"]
        df_automation_systems = benchmark_dataset["df_automation_systems"]
        dynamic_models = [df_static_element_models, df_automation_systems]
        model_mapping = _map_dynamic_models_dynawo(dynamic_models)

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()

        res = sim.run(
            network=pp_net_ieee14,
            model_mapping=model_mapping,
            event_mapping=event_mapping_ieee14,
            timeseries_mapping=variable_mapping_ieee14,
            parameters=param_ieee14,
            report_node=report_node,
        )
        assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)

    def test_wecc_model_mapping(self, unit_test_param):
        """Test the mapping for a WECC wind model."""
        from gridfm_datakit.dynamic.dynawo import _map_dynamic_models_dynawo

        pp_net = pp.network.create_ieee9()

        df_static_element_models = pd.DataFrame.from_records(
            columns=["category_name", "static_id", "parameter_set_id", "model_name"],
            data=[
                (
                    "Wecc",
                    "B1-G",
                    "WECC_WTG4A",
                    "WTG4AWeccCurrentSource",
                ),
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

        dynamic_models = [df_static_element_models, df_automation_systems]
        model_mapping = _map_dynamic_models_dynawo(dynamic_models)

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()
        sim.run(
            network=pp_net,
            model_mapping=model_mapping,
            event_mapping=None,
            timeseries_mapping=None,
            parameters=unit_test_param,
            report_node=report_node,
        )

        report_json = report_node.to_json()
        assert len(_failed_model_instantiations(report_json)) == 0

    def test_overload_management_system_mapping(self, unit_test_param):
        """Test the mapping for an overload_management_system."""
        from gridfm_datakit.dynamic.dynawo import _map_dynamic_models_dynawo

        pp_net = pp.network.create_ieee9()

        df_static_element_models = pd.DataFrame(
            columns=["category_name", "static_id", "parameter_set_id", "model_name"],
        )

        df_automation_systems = pd.DataFrame.from_records(
            columns=[
                "category_name",
                "dynamic_model_id",
                "parameter_set_id",
                "params",
                "model_name",
            ],
            data=[
                (
                    "OverloadManagementSystem",
                    "OMS",
                    "OMS",
                    "controlled_branch=L7-8-0;i_measurement=L7-8-0;i_measurement_side=ONE",
                    "OverloadManagementSystem",
                ),
            ],
        )

        dynamic_models = [df_static_element_models, df_automation_systems]
        model_mapping = _map_dynamic_models_dynawo(dynamic_models)

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()
        sim.run(
            network=pp_net,
            model_mapping=model_mapping,
            event_mapping=None,
            timeseries_mapping=None,
            parameters=unit_test_param,
            report_node=report_node,
        )

        report_json = report_node.to_json()
        assert len(_failed_model_instantiations(report_json)) == 0

    def test_two_level_overload_management_system_mapping(self, unit_test_param):
        """Test the mapping for a two lovel overload management system."""
        from gridfm_datakit.dynamic.dynawo import _map_dynamic_models_dynawo

        pp_net = pp.network.create_ieee9()

        df_static_element_models = pd.DataFrame(
            columns=["category_name", "static_id", "parameter_set_id", "model_name"],
        )

        df_automation_systems = pd.DataFrame.from_records(
            columns=[
                "category_name",
                "dynamic_model_id",
                "parameter_set_id",
                "params",
                "model_name",
            ],
            data=[
                (
                    "TwoLevelOverloadManagementSystem",
                    "TLOMS",
                    "TLOMS",
                    "controlled_branch=L7-8-0;i_measurement_1=L7-8-0;i_measurement_1_side=ONE;i_measurement_2=L7-8-0;i_measurement_2_side=TWO",
                    "TwoLevelOverloadManagementSystem",
                ),
            ],
        )

        dynamic_models = [df_static_element_models, df_automation_systems]
        model_mapping = _map_dynamic_models_dynawo(dynamic_models)

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()
        sim.run(
            network=pp_net,
            model_mapping=model_mapping,
            event_mapping=None,
            timeseries_mapping=None,
            parameters=unit_test_param,
            report_node=report_node,
        )

        report_json = report_node.to_json()
        assert len(_failed_model_instantiations(report_json)) == 0

    def test_tap_changer_mapping(self, unit_test_param):
        """Test the mapping for a tap changer model."""
        from gridfm_datakit.dynamic.dynawo import _map_dynamic_models_dynawo

        pp_net = pp.network.create_ieee9()

        df_static_element_models = pd.DataFrame(
            columns=["category_name", "static_id", "parameter_set_id", "model_name"],
            data=[
                ("LoadOneTransformer", "B5-L", "LoadOneTransfo", "LoadOneTransformer"),
            ],
        )

        df_automation_systems = pd.DataFrame.from_records(
            columns=[
                "category_name",
                "dynamic_model_id",
                "parameter_set_id",
                "params",
                "model_name",
            ],
            data=[
                (
                    "TapChanger",
                    "TapChanger_id",
                    "TapChanger_par",
                    "static_id=B5-L;side=NONE",
                    "TapChangerAutomationSystem",
                ),
            ],
        )

        dynamic_models = [df_static_element_models, df_automation_systems]
        model_mapping = _map_dynamic_models_dynawo(dynamic_models)

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()
        sim.run(
            network=pp_net,
            model_mapping=model_mapping,
            event_mapping=None,
            timeseries_mapping=None,
            parameters=unit_test_param,
            report_node=report_node,
        )

        # note: this test only validates the instantiation
        report_json = report_node.to_json()
        assert len(_failed_model_instantiations(report_json)) == 0

    def test_phase_shifter_blocking_i_mapping(self, unit_test_param):
        """Test the mapping for a phase shifter i and its blocking system."""
        from gridfm_datakit.dynamic.dynawo import _map_dynamic_models_dynawo

        pp_net = pp.network.create_ieee300()

        df_static_element_models = pd.DataFrame(
            columns=["category_name", "static_id", "parameter_set_id", "model_name"],
        )

        df_automation_systems = pd.DataFrame.from_records(
            columns=[
                "category_name",
                "dynamic_model_id",
                "parameter_set_id",
                "params",
                "model_name",
            ],
            data=[
                (
                    "PhaseShifterI",
                    "PSI",
                    "PSI_par",
                    "transformer=T196-2040-1",
                    "PhaseShifterI",
                ),
                (
                    "PhaseShifterBlockingI",
                    "PSBI_id",
                    "PSBI_par",
                    "phase_shifter_id=PSI",
                    "PhaseShifterBlockingI",
                ),
            ],
        )

        dynamic_models = [df_static_element_models, df_automation_systems]
        model_mapping = _map_dynamic_models_dynawo(dynamic_models)

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()
        sim.run(
            network=pp_net,
            model_mapping=model_mapping,
            event_mapping=None,
            timeseries_mapping=None,
            parameters=unit_test_param,
            report_node=report_node,
        )
        # note: this test only validates the instantiation
        report_json = report_node.to_json()
        assert len(_failed_model_instantiations(report_json)) == 0

    def test_phase_shifter_p_mapping(self, unit_test_param):
        """Test the mapping for a phase shifter p system."""
        from gridfm_datakit.dynamic.dynawo import _map_dynamic_models_dynawo

        pp_net = pp.network.create_ieee300()

        df_static_element_models = pd.DataFrame(
            columns=["category_name", "static_id", "parameter_set_id", "model_name"],
        )

        df_automation_systems = pd.DataFrame.from_records(
            columns=[
                "category_name",
                "dynamic_model_id",
                "parameter_set_id",
                "params",
                "model_name",
            ],
            data=[
                (
                    "PhaseShifterP",
                    "PSP",
                    "PSP_par",
                    "transformer=T196-2040-1",
                    "PhaseShifterP",
                ),
            ],
        )

        dynamic_models = [df_static_element_models, df_automation_systems]
        model_mapping = _map_dynamic_models_dynawo(dynamic_models)

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()
        sim.run(
            network=pp_net,
            model_mapping=model_mapping,
            event_mapping=None,
            timeseries_mapping=None,
            parameters=unit_test_param,
            report_node=report_node,
        )
        # note: this test only validates the instantiation
        report_json = report_node.to_json()
        assert len(_failed_model_instantiations(report_json)) == 0


class TestEventMapping:
    """Tests for the event mapper."""

    def test_event_mapping_on_benchmark(
        self,
        pp_net_ieee14,
        model_mapping_ieee14,
        variable_mapping_ieee14,
        param_ieee14,
        df_ref_curves_ieee14,
        benchmark_dataset,
    ):
        """Test the event mapping on benchmark."""
        from gridfm_datakit.dynamic.dynawo import _map_events_dynawo

        df_events = benchmark_dataset["df_events"]
        event_mapping = _map_events_dynawo(df_events)

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()

        res = sim.run(
            network=pp_net_ieee14,
            model_mapping=model_mapping_ieee14,
            event_mapping=event_mapping,
            timeseries_mapping=variable_mapping_ieee14,
            parameters=param_ieee14,
            report_node=report_node,
        )

        assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)

    def test_disconnect_event_mapping(
        self,
        pp_net_ieee14,
        model_mapping_ieee14,
        variable_mapping_ieee14,
        param_ieee14,
    ):
        """Test the mapping for a disconnection event."""
        from gridfm_datakit.dynamic.dynawo import _map_events_dynawo

        df_event = pd.DataFrame.from_records(
            columns=["event_name", "static_id", "start_time", "params"],
            data=[
                ("Disconnect", "_GEN____2_SM", 50, "disconnect_only=;"),
            ],
        )
        event_mapping = _map_events_dynawo(df_event)
        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()

        sim.run(
            network=pp_net_ieee14,
            model_mapping=model_mapping_ieee14,
            event_mapping=event_mapping,
            timeseries_mapping=variable_mapping_ieee14,
            parameters=param_ieee14,
            report_node=report_node,
        )

        report_json = report_node.to_json()
        assert len(_failed_model_instantiations(report_json)) == 0

    def test_active_power_variation_event_mapping(
        self,
        pp_net_ieee14,
        model_mapping_ieee14,
        variable_mapping_ieee14,
        param_ieee14,
    ):
        """Test the mapping for an active power variation event."""
        from gridfm_datakit.dynamic.dynawo import _map_events_dynawo

        df_event = pd.DataFrame.from_records(
            columns=["event_name", "static_id", "start_time", "params"],
            data=[
                ("ActivePowerVariation", "_GEN____2_SM", 50, "delta_p=2"),
            ],
        )
        event_mapping = _map_events_dynawo(df_event)
        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()

        sim.run(
            network=pp_net_ieee14,
            model_mapping=model_mapping_ieee14,
            event_mapping=event_mapping,
            timeseries_mapping=variable_mapping_ieee14,
            parameters=param_ieee14,
            report_node=report_node,
        )

        report_json = report_node.to_json()
        failed = _failed_model_instantiations(report_json)
        assert len(failed) == 0

    def test_reactive_power_variation_event_mapping(
        self,
        pp_net_ieee14,
        variable_mapping_ieee14,
        param_ieee14,
        benchmark_dataset,
    ):
        """Test the mapping for a reactive power variation event."""
        from gridfm_datakit.dynamic.dynawo import _map_events_dynawo
        from gridfm_datakit.dynamic.dynawo import _map_dynamic_models_dynawo

        # drop the dynamic model for _LOAD___2_EC to apply Q variation
        # otherwise a model compatible with Q variation should be mapped.
        # On a copy: the fixture is shared with every other test in this module.
        df_static_elem_dyn_models = benchmark_dataset[
            "df_static_element_dynamic_models"
        ].drop(index=5)
        df_automation_systems = benchmark_dataset["df_automation_systems"]

        df_event = pd.DataFrame.from_records(
            columns=["event_name", "static_id", "start_time", "params"],
            data=[
                ("ReactivePowerVariation", "_LOAD___2_EC", 50, "delta_q=2"),
            ],
        )
        event_mapping = _map_events_dynawo(df_event)

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()

        sim.run(
            network=pp_net_ieee14,
            model_mapping=_map_dynamic_models_dynawo(
                [df_static_elem_dyn_models, df_automation_systems],
            ),
            event_mapping=event_mapping,
            timeseries_mapping=variable_mapping_ieee14,
            parameters=param_ieee14,
            report_node=report_node,
        )

        report_json = report_node.to_json()
        failed = _failed_model_instantiations(report_json)
        assert len(failed) == 0

    def test_node_fault_event_mapping(
        self,
        pp_net_ieee14,
        model_mapping_ieee14,
        variable_mapping_ieee14,
        param_ieee14,
    ):
        """Test the mapping for a node fault event."""
        from gridfm_datakit.dynamic.dynawo import _map_events_dynawo

        df_event = pd.DataFrame.from_records(
            columns=["event_name", "static_id", "start_time", "params"],
            data=[
                ("NodeFault", "_BUS____2_TN", 50, "fault_time=0.2;r_pu=0;x_pu=0.2"),
            ],
        )
        event_mapping = _map_events_dynawo(df_event)
        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()

        sim.run(
            network=pp_net_ieee14,
            model_mapping=model_mapping_ieee14,
            event_mapping=event_mapping,
            timeseries_mapping=variable_mapping_ieee14,
            parameters=param_ieee14,
            report_node=report_node,
        )

        report_json = report_node.to_json()
        failed = _failed_model_instantiations(report_json)
        assert len(failed) == 0

    def test_reference_voltage_variation_event_mapping(
        self,
        pp_net_ieee14,
        model_mapping_ieee14,
        variable_mapping_ieee14,
        param_ieee14,
    ):
        """Test the mapping for a reference voltage variation event."""
        from gridfm_datakit.dynamic.dynawo import _map_events_dynawo

        df_event = pd.DataFrame.from_records(
            columns=["event_name", "static_id", "start_time", "params"],
            data=[
                ("ReferenceVoltageVariation", "_GEN____3_SM", 50, "delta_u=200"),
            ],
        )
        event_mapping = _map_events_dynawo(df_event)
        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()

        sim.run(
            network=pp_net_ieee14,
            model_mapping=model_mapping_ieee14,
            event_mapping=event_mapping,
            timeseries_mapping=variable_mapping_ieee14,
            parameters=param_ieee14,
            report_node=report_node,
        )

        report_json = report_node.to_json()
        failed = _failed_model_instantiations(report_json)
        assert len(failed) == 0


class TestVariableMapping:
    """Tests for the variable mapper."""

    def test_variable_mapping_on_benchmark(
        self,
        pp_net_ieee14,
        model_mapping_ieee14,
        event_mapping_ieee14,
        param_ieee14,
        df_ref_curves_ieee14,
        benchmark_dataset,
    ):
        """Test the variable mapping on benchmark."""
        from gridfm_datakit.dynamic.dynawo import _map_variables_dynawo

        df_variables = benchmark_dataset["df_variables"]
        variable_mapping = _map_variables_dynawo(df_variables)

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()

        res = sim.run(
            network=pp_net_ieee14,
            model_mapping=model_mapping_ieee14,
            event_mapping=event_mapping_ieee14,
            timeseries_mapping=variable_mapping,
            parameters=param_ieee14,
            report_node=report_node,
        )

        assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)

    def test_final_state_value_mapping(
        self,
        pp_net_ieee14,
        model_mapping_ieee14,
        event_mapping_ieee14,
        param_ieee14,
    ):
        """Test the variable mapping for final state value."""
        from gridfm_datakit.dynamic.dynawo import _map_variables_dynawo

        df_variables = pd.DataFrame.from_records(
            columns=["type", "model_id", "variables"],
            data=[
                ("Curve", "_BUS____2_TN", "U_value"),
                ("FinalStateValue", "_BUS____2_TN", "U_value"),
            ],
        )
        variable_mapping = _map_variables_dynawo(df_variables)

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()

        res = sim.run(
            network=pp_net_ieee14,
            model_mapping=model_mapping_ieee14,
            event_mapping=event_mapping_ieee14,
            timeseries_mapping=variable_mapping,
            parameters=param_ieee14,
            report_node=report_node,
        )
        final_value_from_curves = res.curves().iloc[-1].values
        final_value = res.final_state_values().loc[
            "NETWORK__BUS____2_TN_U_value",
            "values",
        ]
        assert final_value == final_value_from_curves
        assert len(res.final_state_values()) == 1

    def test_param_value_containing_equals_sign(self):
        """A param value that itself contains '=' must parse, not raise.

        The parser used an unbounded split('='), so 'k=a=b' produced a 3-element list
        and dict() raised "dictionary update sequence element #0 has length 3".
        """
        from gridfm_datakit.dynamic.dynawo import _get_param_value

        params = "generator=_GEN=3;x_pu=0.1"
        assert _get_param_value(params, "generator") == "_GEN=3"
        assert _get_param_value(params, "x_pu") == 0.1
        # the Disconnect option keeps its "present but empty -> NaN" behaviour
        assert pd.isna(_get_param_value("disconnect_only=;", "disconnect_only"))


class TestDynawoMapping:
    """Tests for the full mapping process."""

    def test_generate_dynawo_mapping_on_benchmark(
        self,
        pp_net_ieee14,
        param_ieee14,
        df_ref_curves_ieee14,
        benchmark_dataset,
    ):
        """Test the full mapping on benchmark."""
        from gridfm_datakit.dynamic import DynamicInputs
        from gridfm_datakit.dynamic.dynawo import generate_dynawo_mappings

        df_static_element_models = benchmark_dataset["df_static_element_dynamic_models"]
        df_automation_systems = benchmark_dataset["df_automation_systems"]
        df_events = benchmark_dataset["df_events"]
        df_variables = benchmark_dataset["df_variables"]

        sim = pp.dynamic.Simulation()
        report_node = pp.report.ReportNode()
        dynamic_inputs = DynamicInputs(
            dynamic_models=[
                df_static_element_models,
                df_automation_systems,
            ],
            events=df_events,
            variables=df_variables,
        )
        dynawo_mapping = generate_dynawo_mappings(dynamic_inputs)

        res = sim.run(
            network=pp_net_ieee14,
            model_mapping=dynawo_mapping.dynamic_model_mapping,
            event_mapping=dynawo_mapping.event_mapping,
            timeseries_mapping=dynawo_mapping.variable_mapping,
            parameters=param_ieee14,
            report_node=report_node,
        )
        assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)
