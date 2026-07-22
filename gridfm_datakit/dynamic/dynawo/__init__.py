"""
Dynaωo-specific submodule for gridfm_datakit dynamic simulation.

Exposes:
- DynawoMappings : dataclass holding the three DataFrames that map directly
  onto pypowsybl.dynamic's add_all_dynamic_mappings / event / curve APIs.
- generate_dynawo_mappings : convert a DynamicInputs into DynawoMappings.
- get_dynawo_simulation_parameters : build a pypowsybl.dynamic.Parameters object
  from the config's dynamic.solver_parameters block.
- get_dynawo_loadflow_parameters : build the pypowsybl.loadflow.Parameters used
  for the balanced initial state, from the config's dynamic.loadflow_parameters
  block.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    LOADFLOW_PARAMETERS_DEFAULTS,
    SIMULATION_PARAMETERS_MAPPING,
)


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
    """Prepares the parameters for Dynawo simulation.

    Raises
    ------
    ValueError
        If a required key is missing, or an unsupported one is present.
    """
    dict_parameters = args.dynamic.solver_parameters.to_dict()

    # Both checks run in the parent process, before any worker is spawned: a bad
    # key raised inside a worker is a bare KeyError that fails every chunk with no
    # hint as to which setting was wrong.
    missing = sorted({"start_time", "stop_time"} - set(dict_parameters))
    if missing:
        raise ValueError(
            f"dynamic.solver_parameters: missing required key(s) {missing}. "
            "Both bound the simulation window, in seconds.",
        )

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
# Public: get load flow parameters for the balanced initial state
# ---------------------------------------------------------------------------


def get_dynawo_loadflow_parameters(args: NestedNamespace) -> pp.loadflow.Parameters:
    """Build the AC load flow parameters for the balanced initial state.

    The defaults deliberately differ from ``powsybl.get_default_lf_params()``,
    which the static pipeline uses. Dynawo initialises each synchronous machine
    from the power flow solution, so the slack must sit on a machine that carries
    a dynamic model: ``slackBusSelectionMode=LARGEST_GENERATOR`` puts it there,
    and ``read_slack_bus``/``write_slack_bus`` keep that choice explicit and
    visible in the network. OpenLoadFlow's own default (MOST_MESHED) can select a
    bus with no generator at all, leaving Dynawo to initialise from a state its
    machine models cannot reproduce.

    Every default is overridable through the optional ``dynamic.loadflow_parameters``
    config block.

    Raises
    ------
    ValueError
        If the config block contains an unsupported key.
    """
    cfg = getattr(getattr(args, "dynamic", None), "loadflow_parameters", None)
    overrides = cfg.to_dict() if cfg is not None else {}

    unknown = sorted(set(overrides) - set(LOADFLOW_PARAMETERS_DEFAULTS))
    if unknown:
        raise ValueError(
            f"dynamic.loadflow_parameters: unsupported key(s) {unknown}. "
            f"Accepted: {sorted(LOADFLOW_PARAMETERS_DEFAULTS)}.",
        )

    params = {**LOADFLOW_PARAMETERS_DEFAULTS, **overrides}
    # provider_parameters is a free-form pass-through to OpenLoadFlow, which only
    # accepts strings.
    provider_parameters = {
        str(key): str(value)
        for key, value in (params["provider_parameters"] or {}).items()
    }

    return pp.loadflow.Parameters(
        distributed_slack=bool(params["distributed_slack"]),
        read_slack_bus=bool(params["read_slack_bus"]),
        write_slack_bus=bool(params["write_slack_bus"]),
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
    # Read-only: every frame written below is a fresh one produced by
    # groupby/get_group/reset_index, so the caller's inputs are never mutated.
    df_dm, df_as = dynamic_models

    # Values were validated against the mapping tables by
    # gridfm_datakit.dynamic._validate_dynawo_values before this point.

    # map the dynamic models
    cats_dm = df_dm["category_name"].unique()
    df_grp_dm = df_dm.groupby("category_name")
    for cat in cats_dm:
        df_cat = (
            df_grp_dm.get_group(cat)[["static_id", "parameter_set_id", "model_name"]]
            .reset_index(drop=True)
            .set_index("static_id")
        )
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
        )
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
    event_mapping = pp.dynamic.EventMapping()
    event_types = events["event_name"].unique()
    df_grp_event = events.groupby("event_name")
    for type_t in event_types:
        df_event_type_t = df_grp_event.get_group(type_t)
        df_event_type_t = (
            df_event_type_t[["static_id", "start_time", "params"]]
            .reset_index(drop=True)
            .set_index("static_id")
        )
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
            # Only non-empty frames are registered: pypowsybl gives no guarantee
            # about an empty event frame, and one of these two is usually empty.
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
    "get_dynawo_loadflow_parameters",
    # data class
    "DynawoMappings",
]
