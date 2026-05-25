# TODO: same as for PowSyBl
from dataclasses import dataclass
import pypowsybl as pp
from typing import Any

from gridfm_datakit.utils.param_handler import NestedNamespace

# from gridfm_datakit.dynamic import DynamicInputs

@dataclass
class DynawoMappings:
    """
    Data class for the mappings of the dynamic inputs for Dynawo.
    3 mappings are needed:
        - dynamic model mapping
        - event mapping
        - variable mapping
    """
    dynamic_model_mapping: Any # TODO: change to Dynawo's mapping 
    event_mapping: Any
    variable_mapping: Any

## Public API

def generate_dynawo_mappings(dynamic_inputs) -> DynawoMappings:
    """Prepares dynamic mappings for Dynawo, using a DynamicInputs object."""
    
    # TODO: do conversion from Dynamic Inputs for real
    dynamic_model_mapping = _map_dynamic_models_dynawo(dynamic_inputs.dynamic_models)
    event_mapping = _map_events_dynawo(dynamic_inputs.events)
    variable_mapping = _map_variables_dynawo(dynamic_inputs.variables)

    return DynawoMappings(
        dynamic_model_mapping=dynamic_model_mapping,
        event_mapping=event_mapping,
        variable_mapping=variable_mapping
    )

def _map_dynamic_models_dynawo(dynamic_models):
    """Map dynamic models inputs to Dynawo format."""

def _map_events_dynawo(events):
    """Map event inputs to Dynawo format."""

def _map_variables_dynawo (variables):
    """Map variable inputs to Dynawo format."""

def prepare_dynawo_parameters(args:NestedNamespace):
    """Prepares the parameters for Dynawo simulation."""
    params = pp.dynamic.Parameters(
        start_time=args.dynamic.start_time,
        stop_time=args.dynamic.stop_time,
        provider_parameters={
            'parametersFile': args.dynamic.param_path,
            'network.parametersFile': args.dynamic.network_param_path,
            'network.parametersId': args.dynamic.network_param_id,
            'solver.type': args.dynamic.solver_type,
            'solver.parametersFile': args.dynamic.solver_param_path,
            'solver.parametersId': args.dynamic.solver_param_id
            }
        )
    return params

__all__ = [
    # primary entry points 
    "generate_dynawo_mappings",
    "prepare_dynawo_parameters",
    # data class
    "DynawoMappings"
]