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
    "category_name",  # str; category of the dynamic model
    "static_id",  # str; pypowsybl element ID to equip with a dynamic model
    "parameter_set_id",  # str; parameter set identifier (references a .par file group)
    "model_name",  # str; Dynawo model name  (e.g. "GeneratorSynchronousThreeWindings")
}

AUTOMATION_SYSTEMS_REQUIRED_COLS = {
    "category_name",  # str; category of the automation system
    "dynamic_model_id",  # str; identifier given to the automation system
    "parameter_set_id",  # str; parameter set identifier
    "params",  # str; parameters that vary according to category. Given as a single string with pattern: "<parameter1>=<value1>;<parameter2>=<value2>;..."
    "model_name",  # str; Dynawo model name (e.g. "UnderVoltage")
}

# events

EVENTS_REQUIRED_COLS = {
    "event_name",  # str; specificy the type of event, options = ['ActivePowerVariation', 'Disconnect', 'NodeFault', 'ReactivePowerVariation', 'ReferenceVoltageVariation']
    "static_id",  # str; pypowsybl element ID to which the event applies
    "start_time",  # str or float; event start time
    "params",  # str; parameters that vary according to event type. Given as a single string with pattern: "<parameter1>=<value1>;<parameter2>=<value2>;..."
}

# variables

VARIABLES_REQUIRED_COLS = {
    "type",  # str; either "Curve" for timeseries or "FinalStateValue" for the final state value only
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
    dynamic_results : pandas.DataFrame
        Curves for one sample, indexed by time: shape **(n_timesteps, n_variables)**,
        one column per output variable. This is the solver's natural orientation
        and is kept as-is through the pipeline.

        Note the persistent store uses the *transposed* orientation:
        ``_save_generated_data`` transposes each sample to (n_variables,
        n_timesteps) and stacks them into a Zarr array of shape
        (n_scenarios, n_variables, n_timesteps).
    report : Any
        Dynamic simulation report including model build-up and problem resolution.
    """

    dynamic_results: Any  # pandas.DataFrame — (n_timesteps, n_variables)
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
        _load_table(dyn_input_cfg.automation_systems_file),
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
            "automation_systems",
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
        dynamic_models = [_normalize_dtypes(df) for df in dynamic_models]
        events = _normalize_dtypes(events)
        variables = _normalize_dtypes(variables)
        _validate_dynawo_values(dynamic_models[1], events, variables)

    return DynamicInputs(
        dynamic_models=dynamic_models,
        events=events,
        variables=variables,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


# Columns whose values are element/model IDs or names, and must be strings:
# a numeric-looking ID (e.g. a bus named "2") would otherwise be read as int and
# fail to match pypowsybl's string IDs. ``start_time`` is the only numeric field.
_STRING_COLS = {
    "category_name",
    "static_id",
    "parameter_set_id",
    "model_name",
    "dynamic_model_id",
    "params",
    "event_name",
    "type",
    "model_id",
    "variables",
}


def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ID/name columns to str and ``start_time`` to float in place.

    Guards against numeric-looking IDs silently becoming ints (which then fail to
    match pypowsybl's string element IDs). Missing string values become "".
    """
    for col in df.columns:
        if col in _STRING_COLS:
            df[col] = df[col].astype("string").fillna("").astype(str)
        elif col == "start_time":
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    return df


def _validate_dynawo_values(
    automation_systems: pd.DataFrame,
    events: pd.DataFrame,
    variables: pd.DataFrame,
) -> None:
    """Validate the *values* of the Dynawo input tables, not just their columns.

    Every value checked here is used later as a key into one of the mapping dicts
    in dynawo/utils.py (or as a branch in _map_variables_dynawo). Without this,
    a single typo surfaces either as a bare ``KeyError`` from deep inside a worker
    process — where it fails every chunk and the user only ever sees "every
    scenario failed" — or, worse, is silently ignored: an unrecognised variable
    ``type`` monitors nothing, so Dynawo returns SUCCESS with empty curves and the
    run only dies much later, in the Zarr writer, after all the compute.

    Raises
    ------
    ValueError
        Naming the offending value, its column, and the accepted values.
    """
    # Imported here rather than at module scope: these are Dynawo-specific and this
    # module is the solver-agnostic layer.
    from gridfm_datakit.dynamic.dynawo.utils import (
        AUTOMATION_SYSTEM_PARAMS_MAPPING,
        EVENT_PARAMS_MAPPING,
        VARIABLE_TYPES,
    )

    def _check_values(df, column, accepted, file_label):
        if column not in df.columns:  # already reported by _check_cols
            return
        unknown = sorted(set(df[column].unique()) - set(accepted))
        if unknown:
            raise ValueError(
                f"{file_label}: unsupported {column} {unknown}. "
                f"Accepted values: {sorted(accepted)}.",
            )

    _check_values(
        automation_systems,
        "category_name",
        AUTOMATION_SYSTEM_PARAMS_MAPPING,
        "automation_systems",
    )
    _check_values(events, "event_name", EVENT_PARAMS_MAPPING, "events")
    _check_values(variables, "type", VARIABLE_TYPES, "variables")

    # The dynamic time-series store is built from "Curve" rows only. With none, the
    # simulation runs, monitors nothing, and returns an empty curves frame — which
    # would otherwise blow up the Zarr writer with an opaque ZeroDivisionError.
    if "type" in variables.columns and "Curve" not in set(variables["type"]):
        raise ValueError(
            "variables: no row of type 'Curve'. The dynamic results store is built "
            "from Curve rows, so at least one is required.",
        )


def _check_cols(df: pd.DataFrame, required: set[str], file_label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {sorted(missing)}. "
            f"Present columns: {list(df.columns)}",
        )


# Delimiters the CSV sniffer is allowed to choose between. Restricting the set is
# what makes sniffing safe: left unconstrained, csv.Sniffer picks any character
# that looks regular, so a single-column file yields a letter ("e") and parses to
# silent garbage. Covers the comma/semicolon convention split plus tabs.
_CSV_DELIMITERS = ",;\t"


def _read_csv_sample(p: Path) -> str:
    """Return whole lines from the head of a CSV, for delimiter sniffing."""
    sample = p.read_text()[:8192]
    # Never end mid-line: a truncated last line can carry an unbalanced quote or a
    # partial field, either of which skews the sniffer.
    cut = sample.rfind("\n")
    return sample[: cut + 1] if cut != -1 else sample


def _load_table(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame.

    CSV delimiters are sniffed rather than assumed, so the files work under both
    the North American (",") and European (";") conventions. The sniffer falls
    back to "," when it cannot decide — which is the case for a header-only or
    single-column file, where there is no delimiter to find.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dynamic input file not found: {path}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        try:
            sep = (
                csv.Sniffer()
                .sniff(
                    _read_csv_sample(p),
                    delimiters=_CSV_DELIMITERS,
                )
                .delimiter
            )
        except csv.Error:
            sep = ","
        return pd.read_csv(p, sep=sep)
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
