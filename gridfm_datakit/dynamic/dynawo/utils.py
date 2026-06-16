AUTOMATION_SYSTEM_PARAMS_MAPPING = {
    'OverloadManagementSystem': ['controlled_branch', 'i_measurement', 'i_measurement_side'],
    'PhaseShifterBlockingI': ['phase_shifter_id'],
    'PhaseShifterI': ['transformer'],
    'PhaseShifterP': ['transformer'],
    'TapChanger': ['static_id', 'side'],
    'TapChangerBlocking': ['rfo_df', 'mp1_df', 'mp2_df', 'mp3_df', 'mp4_df', 'mp5_df'],
    'TwoLevelOverloadManagementSystem': ['controlled_branch', 'i_measurement_1', 'i_measurement_1_side', 'i_measurement_2', 'i_measurement_2_side'],
    'UnderVoltageAutomationSystem': ['generator'],
}

EVENT_PARAM_MAPPING = {
    'ActivePowerVariation': ['delta_p'],
    'Disconnect': ['disconnect_only'],
    'NodeFault': ['fault_time', 'r_pu', 'x_pu'],
    'ReactivePowerVariation': ['delta_q'],
    'ReferenceVoltageVariation': ['dekta_u']
}

SIMULATION_PARAMETERS_MAPPING = {
    'parameters_file': 'parametersFile',
    'network_parameters_file': 'network.parametersFile',
    'network_parameters_id': 'network.parametersId',
    'solver_type': 'solver.type',
    'solver_parameters_file': 'solver.parametersFile',
    'solver_parameters_id': 'solver.parametersId',
    'precision': 'precision',
}