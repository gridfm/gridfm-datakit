AUTOMATION_SYSTEM_PARAMS = {
    'OverloadManagementSystem': ['controlled_branch', 'i_measurement', 'i_measurement_side'],
    'PhaseShifterBlockingI': ['phase_shifter_id'],
    'PhaseShifterI': ['transformer'],
    'PhaseShifterP': ['transformer'],
    'TapChanger': ['static_id', 'side'],
    'TapChangerBlocking': ['rfo_df', 'mp1_df', 'mp2_df', 'mp3_df', 'mp4_df', 'mp5_df'],
    'TwoLevelOverloadManagementSystem': ['controlled_branch', 'i_measurement_1', 'i_measurement_1_side', 'i_measurement_2', 'i_measurement_2_side'],
    'UnderVoltageAutomationSystem': ['generator'],
}

EVENT_MAPPING_PARAM = {
    'ActivePowerVariation': ['delta_p'],
    'Disconnect': ['disconnect_only'],
    'NodeFault': ['fault_time', 'r_pu', 'x_pu'],
    'ReactivePowerVariation': ['delta_q'],
    'ReferenceVoltageVariation': ['dekta_u']
}