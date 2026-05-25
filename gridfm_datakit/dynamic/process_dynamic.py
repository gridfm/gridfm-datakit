"""Module to process dynamic simulations."""

from typing import Any

from gridfm_datakit.network import Network
from gridfm_datakit.dynamic.dynawo.simulate import run_dynawo_simulation, compute_balanced_static_state_dynawo
from gridfm_datakit.process.process_network import init_julia
from gridfm_datakit.powsybl import LoadedNetwork


def process_dynamic_simulations(loaded_network: LoadedNetwork,
                                scenarios,
                                topology_generator,
                                generation_generator,
                                admittance_generator,
                                file_paths,
                                dynamic_params,
                                dynamic_mappings,
                                args):
    """Process dynamic simulations."""
    processed_data = []

    if args.dynamic.solver == 'dynawo':
        # use gridfm's native modules to generate perturbed networks

        jl = init_julia(args.settings.max_iter, file_paths["solver_log_dir"])
        network = loaded_network.gfm_net
        for scenario_index in range(args.load.scenarios):
            network.Pd = scenarios[:, scenario_index, 0]
            network.Qd = scenarios[:, scenario_index, 1]
            perturbed_networks = topology_generator.generate(network)
            perturbed_networks = generation_generator.generate(perturbed_networks)
            perturbed_networks = admittance_generator.generate(perturbed_networks)

            for perturbed_network in (perturbed_networks):
                # process each scenario, in 4 steps:
                # 1. run opf with PowerModels 
                # 2. update powsybl network with opf results
                # 3. run pf on powsybl network to correct solver difference
                # 4. launch dynamic simulations from a balanced powsybl network.
                combined_res = process_single_dynamic_simulation(loaded_network,
                                                                 perturbed_network,
                                                                 dynamic_mappings,
                                                                 dynamic_params,
                                                                 jl,
                                                                 args.dynamic.solver)
                processed_data.append(combined_res)
        return processed_data
    else:
        raise ValueError(
            f"Dynawo is the only supported dynamic solver for now, got {args.dynamic.solver!r} "
            )

def process_single_dynamic_simulation(
        loaded_network: LoadedNetwork,
        perturbed_network: Network,
        dynamic_mappings,
        parameters: Any,
        jl,
        solver='dynawo'
        ):

    # Compute a balanced state before starting dynamic simulation
    # for the dynawo implementation, it comprises an OPF with powermodels and then a PF using powsybl-open load flow.
    pp_net, pf_res = _compute_balanced_static_state(loaded_network, perturbed_network, jl, solver)

    # TODO: this does not work yet as the computation of the balanced static state needs to convert the powsybl network to a gfm network, we lose the IDs.
    # TODO: need to find a way to keep the IDs. Either by updating an initial powsybl network or changing the name of the elements when converting back from a gfm network.
    # run dynamic
    dynamic_res = _run_dynamic_simulation(pp_net, dynamic_mappings, parameters, solver)

    # combine pf_res and dynamic_res
    combined_res =  _combine_pf_and_dyn_res(pf_res, dynamic_res)
    return combined_res

def _compute_balanced_static_state(
        loaded_network:LoadedNetwork,
        perturbed_network: Network,
        jl,
        solver='dynawo',
        ):
    """
    Run opf to reach a balanced state before starting dynamic simulations.
    
    Returns:
        pp_net: a balanced static network. (powsybl representation for dynawo)
        pf_res: formated power flow results for training.
    """
    if solver == 'dynawo':
        pp_net, pf_res = compute_balanced_static_state_dynawo(loaded_network.pp_net, 
                                                              perturbed_network,
                                                              jl)
        return pp_net, pf_res
    else:
        raise ValueError(
            f"Dynawo is the only supported dynamic solver for now, got {solver!r} "
            )


def _run_dynamic_simulation(
        network, # pp_net but maybe another format -> use loaded_net instead? then gotta change the returned format of _compute_balanced_static_state as well
        dynamic_mappings,
        parameters,
        solver,
        ):
    if solver == 'dynawo':
        # TODO: add a check of the network type ? 
        return run_dynawo_simulation(network, dynamic_mappings, parameters)
    else:
        raise ValueError(
            f"Dynawo is the only supported dynamic solver for now, got {solver!r} "
            )

def _combine_pf_and_dyn_res(pf_res, dyn_res):
    """Combines power flow results with dynamic results."""
    # TODO: define the needed formats
    return (pf_res, dyn_res)
