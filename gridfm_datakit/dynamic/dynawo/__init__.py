"""
Dynaωo-specific submodule for gridfm_datakit dynamic simulation.

Exposes:
- DynawoMappings : dataclass holding the three DataFrames that map directly
  onto pypowsybl.dynamic's add_all_dynamic_mappings / event / curve APIs.
- generate_dynawo_mappings : convert a DynamicInputs into DynawoMappings.
- get_dynawo_simulation_parameters : build a pypowsybl.dynamic.Parameters object
  from the config's dynamic.solver_parameters block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# pypowsybl is an optional dependency. Keep importing this module cheap and
# safe even when it is absent (see gridfm_datakit.dynamic.dynawo.api): the type
# hints below are lazy strings (`from __future__ import annotations`) and every
# runtime use of `pp` lives inside a function that only runs with pypowsybl
# present. Importing gridfm_datakit.dynamic.dynawo must never raise ImportError.
try:
    import pypowsybl as pp
except ImportError:  # pragma: no cover - optional dependency
    pp = None

from gridfm_datakit.utils.param_handler import NestedNamespace
from gridfm_datakit.dynamic import DynamicInputs
from gridfm_datakit.dynamic.dynawo.utils import (
    AUTOMATION_SYSTEM_PARAMS_MAPPING,
    EVENT_PARAMS_MAPPING,
    SIMULATION_PARAMETERS_MAPPING,
)

if TYPE_CHECKING:
    pass


@dataclass
class DynawoMappings:
    """
    Dynawo-ready simulation inputs derived from DynamicInputs.
    3 mappings:
        - dynamic model mapping -> for both static equipments and automation systems
        - event mapping -> for events
        - variable mapping -> for the output variables

    Attributes
    ----------
    dynamic_model_mapping : pypowsybl.dynamic.ModelMapping
    event_mapping : pypowsybl.dynamic.EventMapping
    variable_mapping : pypowsybl.dynamic.OutputVariableMapping
    """

    dynamic_model_mapping: pp.dynamic.ModelMapping
    event_mapping: pp.dynamic.EventMapping
    variable_mapping: pp.dynamic.OutputVariableMapping


# ---------------------------------------------------------------------------
# Public: generate mappings compatibles with Dynawo through PyPowSyBl
# ---------------------------------------------------------------------------


def generate_dynawo_mappings(dynamic_inputs: DynamicInputs) -> DynawoMappings:
    """Convert generic DynamicInputs into Dynawo-compatible DynawoMappings.

    The conversion builds the Dynawo-specific mapping objects by parsing the generic
    DynamicInputs dataframes.

    Args
        dynamic_inputs: DynamicInputs, generic inputs loaded by load_raw_inputs().

    Returns
        DynawoMappings: simulation-ready Dynawo mappings
    """

    dynamic_model_mapping = _map_dynamic_models_dynawo(dynamic_inputs.dynamic_models)
    event_mapping = _map_events_dynawo(dynamic_inputs.events)
    variable_mapping = _map_variables_dynawo(dynamic_inputs.variables)

    return DynawoMappings(
        dynamic_model_mapping=dynamic_model_mapping,
        event_mapping=event_mapping,
        variable_mapping=variable_mapping,
    )


# ---------------------------------------------------------------------------
# Public: get simulation parameters for Dynawo
# ---------------------------------------------------------------------------


def get_dynawo_simulation_parameters(args: NestedNamespace) -> pp.dynamic.Parameters:
    """Prepares the parameters for Dynawo simulation."""
    dict_parameters = args.dynamic.solver_parameters.to_dict()

    # An unmapped key would otherwise be a bare KeyError raised inside a worker,
    # failing every chunk with no hint as to which setting was wrong.
    unknown = sorted(
        set(dict_parameters)
        - set(SIMULATION_PARAMETERS_MAPPING)
        - {"start_time", "stop_time"},
    )
    if unknown:
        raise ValueError(
            f"dynamic.solver_parameters: unsupported key(s) {unknown}. Accepted: "
            f"{sorted(SIMULATION_PARAMETERS_MAPPING)} (plus start_time, stop_time).",
        )

    # provider_parameters only accept strings
    provider_parameters = {
        SIMULATION_PARAMETERS_MAPPING[key]: str(value)
        for key, value in dict_parameters.items()
        if (key not in ["start_time", "stop_time"] and value not in ["none", ""])
    }

    return pp.dynamic.Parameters(
        start_time=dict_parameters["start_time"],
        stop_time=dict_parameters["stop_time"],
        provider_parameters=provider_parameters,
    )


# ---------------------------------------------------------------------------
# Private: mappers
# ---------------------------------------------------------------------------


def _map_dynamic_models_dynawo(
    dynamic_models: list[pd.DataFrame],
) -> pp.dynamic.ModelMapping:
    """Maps dynamic models from DynamicInputs to pypowsybl.dynamic.ModelMapping."""
    # initialize a dynamic model mapping instance
    dynamic_model_mapping = pp.dynamic.ModelMapping()

    # dynamic_models is a list of two pandas dataframes.
    # The first dataframe equips the grid elements with dynamic models
    # The second dataframe introduces automation systems that dynamiclly affect the grid behavior
    # Not only the separation is based the difference in nature but also more practically
    # the automation systems are parsed differently in Dynawo.
    # Plain unpack: the mappers below never write to these frames (every write
    # targets a fresh frame produced by groupby/get_group/reset_index). A
    # list.copy() here would only copy the list, not the DataFrames, so it looked
    # protective while protecting nothing.
    df_dm, df_as = dynamic_models

    # TODO: add validation test for the inputs and flag invalid entries.

    # map the dynamic models
    cats_dm = df_dm["category_name"].unique()
    df_grp_dm = df_dm.groupby("category_name")
    for cat in cats_dm:
        df_cat = (
            df_grp_dm.get_group(cat)[["static_id", "parameter_set_id", "model_name"]]
            .reset_index(drop=True)
            .set_index("static_id")
        )  # TODO: validat with Youssouf, whether we should put the column names here or put it in a utils with something like DYNAWO_DYNAMIC_MAPPING_COLUMNS
        dynamic_model_mapping.add_dynamic_model(category_name=cat, df=df_cat)

    # map the automation systems
    cats_as = df_as["category_name"].unique()
    df_grp_as = df_as.groupby("category_name")
    for cat in cats_as:
        df_cat = (
            df_grp_as.get_group(cat)[
                ["dynamic_model_id", "parameter_set_id", "params", "model_name"]
            ]
            .reset_index(drop=True)
            .set_index("dynamic_model_id")
        )  # TODO: same as above
        param_keywords = AUTOMATION_SYSTEM_PARAMS_MAPPING[cat]
        for keyword in param_keywords:
            df_cat[keyword] = df_cat["params"].map(
                lambda x: _get_param_value(x, keyword),
            )
        df_cat = df_cat[["parameter_set_id"] + param_keywords + ["model_name"]]
        dynamic_model_mapping.add_dynamic_model(category_name=cat, df=df_cat)
    return dynamic_model_mapping


def _map_events_dynawo(events: pd.DataFrame) -> pp.dynamic.EventMapping:
    """Maps the event inputs to Dynawo format."""
    # TODO: validate the inputs events
    # map the events
    event_mapping = pp.dynamic.EventMapping()
    event_types = events["event_name"].unique()
    df_grp_event = events.groupby("event_name")
    for type_t in event_types:
        df_event_type_t = df_grp_event.get_group(type_t)
        df_event_type_t = (
            df_event_type_t[["static_id", "start_time", "params"]]
            .reset_index(drop=True)
            .set_index("static_id")
        )  # TODO: same above
        param_keywords = EVENT_PARAMS_MAPPING[type_t]
        for k in param_keywords:
            df_event_type_t[k] = df_event_type_t["params"].map(
                lambda x: _get_param_value(x, k),
            )

        # specific case for Disconnect which accepts its only parameter 'Disconnect' as an option
        if type_t == "Disconnect":
            df_without_option = df_event_type_t[
                df_event_type_t["disconnect_only"].isna()
            ]
            df_without_option = df_without_option[["start_time"]]

            df_with_option = df_event_type_t[df_event_type_t["disconnect_only"].notna()]
            df_with_option = df_with_option[["start_time", "disconnect_only"]]
            # Skip the empty side: in practice every Disconnect either sets the
            # option or none do, so one of these frames is always empty. Passing an
            # empty frame happens to work today, but nothing guarantees it.
            if not df_without_option.empty:
                event_mapping.add_event_model(event_name=type_t, df=df_without_option)
            if not df_with_option.empty:
                event_mapping.add_event_model(event_name=type_t, df=df_with_option)
        else:
            df_event_type_t = df_event_type_t[["start_time"] + param_keywords]
            event_mapping.add_event_model(event_name=type_t, df=df_event_type_t)
    return event_mapping


def _map_variables_dynawo(variables: pd.DataFrame) -> pp.dynamic.OutputVariableMapping:
    """Map variable inputs to Dynawo format."""
    # TODO: add input validation test
    variable_mapping = pp.dynamic.OutputVariableMapping()
    variable_types = variables["type"].unique()
    df_grp_type = variables.groupby("type")
    for type_t in variable_types:
        df_var_type_t = df_grp_type.get_group(type_t)
        if type_t == "Curve":
            for _, row in df_var_type_t.iterrows():
                variable_mapping.add_curves(
                    model_id=row["model_id"],
                    variables=row["variables"],
                )
        elif type_t == "FinalStateValue":
            for _, row in df_var_type_t.iterrows():
                variable_mapping.add_final_state_values(
                    model_id=row["model_id"],
                    variables=row["variables"],
                )
    return variable_mapping


# ---------------------------------------------------------------------------
# Private: parse the string for the "param" columns of some dynamic inputs: automation systems and events
# ---------------------------------------------------------------------------


def _get_param_value(params, keyword):
    """Gets the value associated to a keyword from the parameters.

    This helper is necessary to handle automation system mapping and event mapping,
    which have different parameters depending on the category of system/event.
    Check gridfm_datakit.dynamic.dynawo.utils for the accepted parameters.

    Pattern: "key1=value1;key2=value2;..."
    """

    # maxsplit=1: only the FIRST "=" separates key from value, so a value that
    # itself contains "=" parses instead of raising ("too many values to unpack").
    pairs = dict(pair.split("=", 1) for pair in params.split(";") if "=" in pair)
    # specific case for the "Disconnect" event, the sole one to have an optional parameter
    if keyword == "disconnect_only" and pairs.get(keyword) == "":
        return np.nan
    return pairs.get(keyword)


__all__ = [
    # primary entry points
    "generate_dynawo_mappings",
    "get_dynawo_simulation_parameters",
    # data class
    "DynawoMappings",
]
