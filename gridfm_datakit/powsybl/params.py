"""Default load flow parameters for power system simulations."""

from .api import pypowsybl as pp


def get_default_lf_params():
    """
    Get default load flow parameters for Open Load Flow solver.

    Returns:
        pp.loadflow.Parameters: Default configuration
    """

    # Deactivating slack distribution to be coherent with existing gridfm_datakit's workflow.
    # If a single slack is specified, it will be used.
    # Otherwise, (e.g. a network initially configured for distributed slack),
    # the bus with the largest generator will be selected as single slack.
    
    # By default, the most meshed bus would be selected as slack to ease branch-related convergence.
    # The power flow result processing attaches the slack result to a generator's output, 
    # hence the choice of the bus with the largest generator as slack.
    return pp.loadflow.Parameters(distributed_slack=False,
                                  read_slack_bus=True,
                                  write_slack_bus=True,
                                  provider_parameters={
                                      'slackBusSelectionMode': 'LARGEST_GENERATOR' # default: MOST_MESHED
                                  })
