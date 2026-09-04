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

    # By default, powsybl selects the most meshed bus as slack to ease branch-related convergence.
    # It could be a bus without any generator.
    # The gridfm's power flow data processing expects the slack to be associated with a generator,
    # hence the choice of the bus with the largest generator as slack to avoid conflict.
    provider_params = {
        "slackBusSelectionMode": "LARGEST_GENERATOR",  # default: MOST_MESHED
    }
    return pp.loadflow.Parameters(
        distributed_slack=False,
        read_slack_bus=True,
        write_slack_bus=True,
        provider_parameters=provider_params,
    )
