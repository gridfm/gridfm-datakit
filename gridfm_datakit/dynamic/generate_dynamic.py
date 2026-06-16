"""Module to generate dynamic simulation data."""
import numpy as np
from typing import Any, Dict, Union, Tuple

from gridfm_datakit.dynamic import(
    load_raw_inputs, 
)

from gridfm_datakit.dynamic.dynawo import(
    get_dynawo_simulation_parameters,
    generate_dynawo_mappings,
)

from gridfm_datakit.generate import (
    _setup_environment,
)

from gridfm_datakit.perturbations.load_perturbation import(
    load_scenarios_to_df,
    plot_load_scenarios_combined,
)

import gridfm_datakit.powsybl as powsybl

from gridfm_datakit.dynamic.process_dynamic import process_dynamic_simulations

from gridfm_datakit.utils.param_handler import (
    NestedNamespace,
    get_load_scenario_generator,
    initialize_admittance_generator,
    initialize_generation_generator,
    initialize_topology_generator
)

def generate_dynamic_data(
        config: Union[str, Dict[str, Any], NestedNamespace]
        ) -> Dict[str, str]:
    
    # TODO: setup logs
    # TODO: revoir le loading du réseau; on devrait utiliser un réseau powsybl comme réseau principal et non gfm. Sinon on perd de l'info.
    
    # setup static environment TODO: reusing _setup_environment from generate.py, good idea? 
    args, base_path, file_paths, seed = _setup_environment(config)
    # prepare network and scenarios
    loaded_network, scenarios = _prepare_network_and_scenarios_dynamic(args, file_paths, seed)

    # prepare randomization generators
    net = loaded_network.gfm_net
    topology_genrator = initialize_topology_generator(args.topology_perturbation, net)
    generation_generator = initialize_generation_generator(args.generation_perturbation, net)
    admittance_generator = initialize_admittance_generator(args.admittance_perturbation, net)

    # get dynamic simulation parameters environment
    dynamic_params = _get_simulation_parameters(args)

    # prepare dynamic inputs
    dynamic_mappings = _load_and_prep_dynamic_mappings(args)

    # process scenarios (randomization -> opf -> pf -> dynamic -> post-process)
    processed_data = process_dynamic_simulations(
        loaded_network,
        scenarios,
        topology_genrator,
        generation_generator,
        admittance_generator,
        file_paths,
        dynamic_params,
        dynamic_mappings,
        args,
        )

    # save results
    _save_generated_data(processed_data)

    return file_paths

def _prepare_network_and_scenarios_dynamic(
    args: NestedNamespace,
    file_paths: Dict[str, str],
    seed: int,
) -> Tuple[Any, np.ndarray]:
    """
    Prepare the network and generate load scenarios.

    Args:
        args: Configuration object
        file_paths: Dictionary of file paths
        seed: Global random seed for reproducibility.

    Returns:
        Tuple[LoadedNetwork, scenarios]
    """

    if args.network.source == "powsybl":
        loaded_net = powsybl.load_net(args.network.file)
        net = loaded_net.gfm_net
    else:
        raise ValueError("Invalid grid source! Powsybl is the only supported source for dynamic simulation")

    # Generate load scenarios
    load_scenario_generator = get_load_scenario_generator(args.load)
    scenarios = load_scenario_generator(
        net,
        args.load.scenarios,
        file_paths["scenarios_log"],
        max_iter=args.settings.max_iter,
        seed=seed,
    )
    scenarios_df = load_scenarios_to_df(scenarios)
    scenarios_df.to_parquet(file_paths["scenarios"], index=False, engine="pyarrow")
    if net.buses.shape[0] <= 100:
        plot_load_scenarios_combined(scenarios_df, file_paths["scenarios_plot"])
    else:
        print("Skipping plot of scenarios for large networks (number of buses > 100)")

    return loaded_net, scenarios

def _load_and_prep_dynamic_mappings(args: NestedNamespace):
    """Loads and converts raw inputs into dynamic mappings."""
    dynamic_inputs = load_raw_inputs(args)
    if args.dynamic.dynamic_solver == 'dynawo':
        return generate_dynawo_mappings(dynamic_inputs)
    else:
        raise ValueError(
            f"Dynawo is the only supported dynamic solver for now, got {args.dynamic.solver!r} "
            )

def _get_simulation_parameters(args: NestedNamespace):
    """Prepares dynamic simulation parameters."""
    if args.dynamic.solver == 'dynawo':
        return get_dynawo_simulation_parameters(args)
    else:
        raise ValueError(
            f"Dynawo is the only supported dynamic solver for now, got {args.dynamic.solver!r} "
            )
        
def _save_generated_data (processed_data):
    """Save the generated data."""
    # TODO: code this function properly
    return None

