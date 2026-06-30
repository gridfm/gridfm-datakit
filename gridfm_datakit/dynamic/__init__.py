"""
Dynamic simulation module for gridfm_datakit.

Provides a solver-agnostic layer for dynamic power system simulation data
generation. The first (and currently only) concrete backend is Dynaωo,
accessed through pypowsybl.dynamic.

Only the generic contracts (DynamicInputs, DynamicResults) and the
load_raw_inputs loader live here. Solver-specific logic is in submodules
(e.g. gridfm_datakit/dynamic/dynawo/).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gridfm_datakit.utils.param_handler import NestedNamespace

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Column schemas for DynamicInputs DataFrames
# ---------------------------------------------------------------------------
# These are the minimum required columns validated by load_raw_inputs when
# dynamic_solver == "dynawo".

# WARNING: for the "static_id", please refer to the IDs given in the network file
# powsybl.network.get_xxx functions may return a slightly contaminated IDs (typically with "_")
# these IDs should not be used, which could lead to errors bulding up the dynamic model.

# The user can check the successful model build-up by reading report_node, included in the DynamicOutputs
# Advanced usage: it is possible to adjust report verbosity through simulation parameter "log.levelFilter"

# dynamic_models (divided into two dataframes static_element_dynamic_models and automation_systems )

STATIC_ELEMENT_DYNAMIC_MODELS_REQUIRED_COLS = {
    "category_name", # str; category of the dynamic model
    "static_id",  # str; pypowsybl element ID to equip with a dynamic model
    "parameter_set_id", # str; parameter set identifier (references a .par file group)
    "model_name",  # str; Dynawo model name  (e.g. "GeneratorSynchronousThreeWindings")
}

AUTOMATION_SYSTEMS_REQUIRED_COLS = {
    "category_name", # str; category of the automation system
    "dynamic_model_id", # str; identifier given to the automation system
    "parameter_set_id", # str; parameter set identifier
    "params", # str; parameters that vary according to category. Given as a single string with pattern: "<parameter1>=<value1>;<parameter2>=<value2>;..."
    "model_name", # str; Dynawo model name (e.g. "UnderVoltage") 
}

# events

EVENTS_REQUIRED_COLS = {
    "event_name", # str; specificy the type of event, options = ['ActivePowerVariation', 'Disconnect', 'NodeFault', 'ReactivePowerVariation', 'ReferenceVoltageVariation']
    "static_id",  # str; pypowsybl element ID to which the event applies
    "start_time",  # str or float; event start time
    "params",  # str; parameters that vary according to event type. Given as a single string with pattern: "<parameter1>=<value1>;<parameter2>=<value2>;..."
}

# variables

VARIABLES_REQUIRED_COLS = {
    "type", # str; either "Curve" for timeseries or "FinalStateValue" for the final state value only
    "model_id",  # str; identifier of the monitored element
    "variables",  # str or list[str]; variables to monitor (check Dynawo dynamic model's description for the list of available variables per model)
}


# ---------------------------------------------------------------------------
# Generic data contracts
# ---------------------------------------------------------------------------


@dataclass
class DynamicInputs:
    """
    Solver-agnostic container for dynamic simulation inputs.

    All three attributes are pandas DataFrames or list of pandas DataFrames so they remain compatible with
    pypowsybl.dynamic's native input format and are easy to inspect or
    serialize for debugging.

    Attributes
    ----------
    dynamic_models : list[pd.DataFrame]
        List of 2 pandas DataFrames. 
        First one is for the dynamic models that equip static elements:
            One row per network element to be equipped with a dynamic model.
            Required columns: category_name, static_id, parameter_set_id, model_name
            Note: unequipped static element will be given a default model
        Second one is for the automation systems:
            One row per automation system
            Required columns: category_name, dynamic_model_id, parameter_set_id, params, model_name 
    events : pd.DataFrame
        One row per event in the simulation sequence.
        Required columns: event_name, static_id, start_time, params
    variables : pd.DataFrame
        One row per monitored output variables (curve or final state value).
        Required columns: type, model_id, variables
    """

    dynamic_models: list[pd.DataFrame]
    events: pd.DataFrame
    variables: pd.DataFrame

@dataclass
class DynamicResults:
    """
    Solver-agnostic container for dynamic simulation outputs.

    Attributes
    ----------
    dynamic_results : zarr.Array or zarr.Group
        Time-series output shaped (n_variables, n_timesteps) per scenario.
        Stored as an in-memory Zarr array during per-scenario processing;
        written to a persistent Zarr store by _save_generated_data.
    report : Any
        Dynamic simulation report including model build-up and problem resolution.
    """
    dynamic_results: Any # zarr.Array or zarr.Group — (n_variables, n_timesteps)
    report: Any


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_raw_inputs(
          args: NestedNamespace,
          ) -> DynamicInputs:
    """Load dynamic simulation inputs from CSV files declared in the config.

    Reads the four CSV (or Parquet) files listed under config.dynamic and
    returns a DynamicInputs instance. When dynamic_solver == "dynawo", the
    minimum required columns for each DataFrame are validated.

    Args
    ----
    args : NestedNamespace
        Configuration object. Must have a ``dynamic`` attribute with:
        - static_element_dynamic_models_file     : path to the models CSV
        - automation_systems_file                : path to the automation systems CSV
        - events_file                            : path to the events CSV
        - variables_file                         : path to the variables CSV
        - dynamic_solver                         : solver name ("dynawo" or future alternatives)

    Returns
    -------
    DynamicInputs

    Raises
    ------
    FileNotFoundError
        If any of the four input files is missing.
    ValueError
        If required columns are absent from a DataFrame (Dynawo solver only).
    TypeError
        If any of the input files is not of CSV or Parquet foramt.
    """
    dyn_input_cfg = args.dynamic.input_files

    dynamic_models = [
        _load_table(dyn_input_cfg.static_element_dynamic_models_file),
        _load_table(dyn_input_cfg.automation_systems_file)
    ]

    events = _load_table(dyn_input_cfg.events_file)
    variables = _load_table(dyn_input_cfg.variables_file)

    solver = getattr(args.dynamic, "dynamic_solver", "dynawo")
    if solver == "dynawo":
        _check_cols(
            dynamic_models[0],
            STATIC_ELEMENT_DYNAMIC_MODELS_REQUIRED_COLS,
            "static_element_dynamic_models",
        )
        _check_cols(
            dynamic_models[1],
            AUTOMATION_SYSTEMS_REQUIRED_COLS,
            "automation_systems"
        )
        _check_cols(
            events,
            EVENTS_REQUIRED_COLS,
            "events",
        )
        _check_cols(
            variables,
            VARIABLES_REQUIRED_COLS,
            "variables",
        )

    return DynamicInputs(
            dynamic_models=dynamic_models,
            events=events,
            variables=variables
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_cols(df: pd.DataFrame, required: set[str], file_label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {sorted(missing)}. "
            f"Present columns: {list(df.columns)}",
        )

def _load_table(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dynamic input file not found: {path}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix.lower() == ".csv": # TODO: validate with Youssouf. To handle convention difference for CSV between North America and Europe, use sniffer
        with open(p) as f:
            dialect = csv.Sniffer().sniff(f.read(1024))
        return pd.read_csv(p, sep=dialect.delimiter)
    else:
        raise TypeError(f"A csv or parquet file is expected, instead received {p}")


__all__ = [
    "DynamicInputs",
    "DynamicResults",
    "load_raw_inputs",
    "STATIC_ELEMENT_DYNAMIC_MODELS_REQUIRED_COLS",
    "AUTOMATION_SYSTEMS_REQUIRED_COLS",
    "EVENTS_REQUIRED_COLS",
    "VARIABLES_REQUIRED_COLS",
]
