# TODO: same as for PowSyBl
from dataclasses import dataclass
import numpy as np
import pypowsybl as pp
from typing import Any

from gridfm_datakit.utils.param_handler import NestedNamespace
from gridfm_datakit.dynamic.dynawo.utils import AUTOMATION_SYSTEM_PARAMS, EVENT_MAPPING_PARAM

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
    
    dynamic_model_mapping = _map_dynamic_models_dynawo(dynamic_inputs.dynamic_models)
    event_mapping = _map_events_dynawo(dynamic_inputs.events)
    variable_mapping = _map_variables_dynawo(dynamic_inputs.variables)

    return DynawoMappings(
        dynamic_model_mapping=dynamic_model_mapping,
        event_mapping=event_mapping,
        variable_mapping=variable_mapping
    )

def _map_dynamic_models_dynawo(dynamic_models):
    """Maps dynamic models inputs to Dynawo format."""
    # initialize a dynamic model mapping instance
    dynamic_model_mapping = pp.dynamic.ModelMapping()
    
    # dynamic_models is a list of two pandas dataframes.
    # The first dataframe equips the grid elements with dynamic models
    # The second dataframe introduces automation systems that dynamiclly affect the grid behavior
    # Not only the separation is based the difference in nature but also more practically
    # the automation systems are parsed differently in Dynawo.
    df_dm, df_as = dynamic_models.copy()
    
    # TODO: add validation test for the inputs and flag invalid entries.

    # map the dynamic models
    cats_dm = df_dm['category_name'].unique()
    df_grp_dm =  df_dm.groupby('category_name')
    for cat in cats_dm:
        df_cat = df_grp_dm.get_group(cat)[['static_id', 'parameter_set_id', 'model_name']].reset_index(drop=True).set_index('static_id') # TODO: validat with Youssouf, whether we should put the column names here or put it in a utils with something like DYNAWO_DYNAMIC_MAPPING_COLUMNS
        dynamic_model_mapping.add_dynamic_model(category_name=cat, df=df_cat)
    
    # map the automation systems
    cats_as = df_as['category_name'].unique()
    df_grp_as = df_as.groupby('category_name')
    for cat in cats_as:
        df_cat = df_grp_as.get_group(cat)[['dynamic_model_id', 'parameter_set_id', 'params', 'model_name']].reset_index(drop=True).set_index('dynamic_model_id') # TODO: same as above
        param_keywords = AUTOMATION_SYSTEM_PARAMS[cat]
        for keyword in param_keywords:
            df_cat[keyword] = df_cat['params'].map(lambda x: _get_param_value(x, keyword))
        df_cat = df_cat[['parameter_set_id'] + param_keywords + ['model_name']]
        dynamic_model_mapping.add_dynamic_model(category_name=cat, df=df_cat)
    return dynamic_model_mapping

def _map_events_dynawo(events):
    """Maps the event inputs to Dynawo format."""
    # TODO: validate the inputs events
    # map the events
    event_mapping = pp.dynamic.EventMapping()
    event_types = events['event_name'].unique()
    df_grp_event = events.groupby('event_name')
    for type_t in event_types:
        df_event_type_t = df_grp_event.get_group(type_t)
        df_event_type_t = df_event_type_t[['static_id', 'start_time', 'params']].reset_index(drop=True).set_index('static_id') # TODO: same above
        param_keywords = EVENT_MAPPING_PARAM[type_t]
        for k in param_keywords:
            df_event_type_t[k] = df_event_type_t['params'].map(lambda x: _get_param_value(x, k))
        
        # specific case for Disconnect which accepts its only parameter 'Disconnect' as an option
        if type_t == 'Disconnect':
            df_without_option = df_event_type_t[df_event_type_t['disconnect_only'].isna()]
            df_without_option = df_without_option[['start_time']]

            df_with_option = df_event_type_t[df_event_type_t['disconnect_only'].notna()]
            df_with_option = df_with_option[['start_time', 'disconnect_only']]
            event_mapping.add_event_model(event_name=type_t, df=df_without_option)
            event_mapping.add_event_model(event_name=type_t, df=df_with_option)
        else:
            df_event_type_t = df_event_type_t[['start_time'] +param_keywords]
            event_mapping.add_event_model(event_name=type_t, df=df_event_type_t)
    return event_mapping

def _map_variables_dynawo (variables):
    """Map variable inputs to Dynawo format."""
    # TODO: add input validation test
    variable_mapping = pp.dynamic.OutputVariableMapping()
    variable_types = variables['type'].unique()
    df_grp_type = variables.groupby('type')
    for type_t in variable_types:
        df_var_type_t = df_grp_type.get_group(type_t)
        if type_t == 'Curve':
            for _, row in df_var_type_t.iterrows():
                variable_mapping.add_curves(model_id=row['model_id'], variables=row['variables'])
        elif type_t == 'FinalStateValue':
                variable_mapping.add_final_state_values(model_id=row['model_id'], variables=row['variables'])
    return variable_mapping


def _get_param_value(params, keyword):
    """Gets the value associated to a keyword from the params."""
    pairs = dict(pair.split("=") for pair in params.split(";") if "=" in pair)
    if keyword == 'disconnect_only' and pairs.get(keyword)=='':
        return np.nan
    return pairs.get(keyword)

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