"""Default load flow parameters for power system simulations."""

import pypowsybl as pp

def get_default_lf_parameters():
    """
    Get default load flow parameters for Open Load Flow solver.

    Returns:
        pp.loadflow.Parameters: Default configution
    """

    # return pp.loadflow.Parameters(
    #     distributed_slack=False,
    #     use_reactive_limits=False,
    #     provider_parameters={
    #         'useActiveLimits': 'false',
    #         'maxNewtonRaphsonIterations': '200',
    #         "reportedFeatures": "NEWTON_RAPHSON_LOAD_FLOW",
    #         "voltageInitMode": "DC_VALUES",
    #         "stateVectorScalingMode": "MAX_VOLTAGE_CHANGE",
    #     }
    # )
    return pp.loadflow.Parameters(
        distributed_slack=False
    )
