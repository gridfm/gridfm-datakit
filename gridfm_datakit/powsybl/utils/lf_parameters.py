"""Default load flow parameters for power system simulations."""

import pypowsybl as pp

def get_default_lf_parameters():
    """
    Get default load flow parameters for Open Load Flow solver.

    Returns:
        pp.loadflow.Parameters: Default configution
    """

    return pp.loadflow.Parameters(
        distributed_slack=False # conform with Power Model behavior;
    )
