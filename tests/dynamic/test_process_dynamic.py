"""Tests for process_dynamic.py"""
import os

import numpy as np


from gridfm_datakit.dynamic.dynawo import generate_dynawo_mappings, get_dynawo_simulation_parameters
from gridfm_datakit.dynamic import load_raw_inputs
from gridfm_datakit.generate import _setup_environment, _prepare_network_and_scenarios
from gridfm_datakit.powsybl import load_net


def test_process_single_dynamic_simulation(config_ieee14):

    from gridfm_datakit.dynamic.process_dynamic import process_single_dynamic_simulation
    from gridfm_datakit.generate import init_julia
    
    config = config_ieee14

    args, _, _,_ = _setup_environment(config)
    dynamic_inputs = load_raw_inputs(args)
    dynawo_mappings = generate_dynawo_mappings(dynamic_inputs)
    simulation_parameters = get_dynawo_simulation_parameters(args)
    net=load_net(args.network.file)

    gfm_net = net.gfm_net
    scenarios = np.zeros((len(gfm_net.Qd), 1, 2))
    scenarios[:, 0, 0] = gfm_net.Pd
    scenarios[:, 0, 1] = gfm_net.Qd

    
    julia = init_julia(200)

    res_dict = process_single_dynamic_simulation(pp_net=net.pp_net,
                                  gfm_net=net.gfm_net,
                                  scenarios=scenarios,
                                  scenario_index=0,
                                  p2g_maps=net.mapping_p2g,
                                  dynamic_mappings=dynawo_mappings,
                                  dynamic_solver_params=simulation_parameters,
                                  dynamic_solver="dynawo",
                                  julia=julia
                                  )
    assert len(res_dict) == 3


def test_process_dynamic_simulation(config_ieee14):

    from gridfm_datakit.dynamic.process_dynamic import process_dynamic_simulations
    
    config = config_ieee14
    
    args, _, file_paths, seed= _setup_environment(config)
    gfm_net, scenarios, meta = _prepare_network_and_scenarios(args, file_paths, seed)
    dynamic_inputs = load_raw_inputs(args)

    res_dict = process_dynamic_simulations(
        network_path=str(meta.get('network_path')),
        scenarios=scenarios,
        dynamic_inputs=dynamic_inputs,
        dynamic_solver='dynawo',
        config=args,
        error_log_file='.',
        seed=seed,
    )
    assert len(res_dict) == args.load.scenarios