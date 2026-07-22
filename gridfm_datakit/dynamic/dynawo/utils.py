AUTOMATION_SYSTEM_PARAMS_MAPPING = {
    "OverloadManagementSystem": [
        "controlled_branch",
        "i_measurement",
        "i_measurement_side",
    ],
    "PhaseShifterBlockingI": ["phase_shifter_id"],
    "PhaseShifterI": ["transformer"],
    "PhaseShifterP": ["transformer"],
    "TapChanger": ["static_id", "side"],
    "TapChangerBlocking": ["rfo_df", "mp1_df", "mp2_df", "mp3_df", "mp4_df", "mp5_df"],
    "TwoLevelOverloadManagementSystem": [
        "controlled_branch",
        "i_measurement_1",
        "i_measurement_1_side",
        "i_measurement_2",
        "i_measurement_2_side",
    ],
    "UnderVoltageAutomationSystem": ["generator"],
}

EVENT_PARAMS_MAPPING = {
    "ActivePowerVariation": ["delta_p"],
    "Disconnect": ["disconnect_only"],
    "NodeFault": ["fault_time", "r_pu", "x_pu"],
    "ReactivePowerVariation": ["delta_q"],
    "ReferenceVoltageVariation": ["delta_u"],
}

# The two accepted values of the "type" column in variables.csv. "Curve" rows
# become time-series (the Zarr store); "FinalStateValue" rows only yield a final
# value. Anything else is a typo — see _validate_dynawo_values.
VARIABLE_TYPES = ["Curve", "FinalStateValue"]

# Defaults for the AC load flow that produces the balanced initial state, and the
# full set of keys accepted under the optional "dynamic.loadflow_parameters"
# config block. See get_dynawo_loadflow_parameters for why these differ from the
# static pipeline's get_default_lf_params().
LOADFLOW_PARAMETERS_DEFAULTS = {
    "distributed_slack": False,
    "read_slack_bus": True,
    "write_slack_bus": True,
    "provider_parameters": {"slackBusSelectionMode": "LARGEST_GENERATOR"},
}

SIMULATION_PARAMETERS_MAPPING = {
    "parameters_file": "parametersFile",
    "network_parameters_file": "network.parametersFile",
    "network_parameters_id": "network.parametersId",
    "solver_type": "solver.type",
    "solver_parameters_file": "solver.parametersFile",
    "solver_parameters_id": "solver.parametersId",
    "precision": "precision",
}
