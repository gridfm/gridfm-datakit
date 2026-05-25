"""General module for dynamic simulations."""

from dataclasses import dataclass
import pandas as pd
from typing import Any

from gridfm_datakit.utils.param_handler import NestedNamespace

@dataclass
class DynamicInputs:
    """
    Data class for dynamic inputs.
    3 categories of inputs:
        - dynamic models
        - events
        - variables
    """
    dynamic_models: Any
    events: Any
    variables: Any


@dataclass
class DynamicResults:
    """
    Data class for dynamic simulation results:
        - dynamic output
        - reporting
    Currently just a wrapper of the pipeline based on Dynawo.
    """
    dynamic_results: Any
    report: Any


def load_raw_inputs(
          args: NestedNamespace,
          input_format='dynawo'
          ) -> DynamicInputs:
    """Load dynamic inputs into a DynamicInputs object."""
    if input_format == 'dynawo':
        # TODO: recuperate the data from csv files.
        dynamic_models = [_load_dynamic_models(args.dynamic.input_paths.dynamic_models), 
                          _load_automation_systems(args.input_paths.automation_systems)
                          ]
        events = _load_events(args.input_paths.events)
        variables = _load_variables(args.input_paths.variables)

    return DynamicInputs(
            dynamic_models=dynamic_models,
            events=events,
            variables=variables
    )

def _load_dynamic_models(
        path_dynamic_models
        ) -> pd.DataFrame:
    return pd.read_csv(path_dynamic_models)

def _load_automation_systems(
        path_automation_systems
        ) -> pd.DataFrame:
    return pd.read_csv(path_automation_systems)

def _load_events(
        path_events
        ) -> pd.DataFrame:
    return pd.read_csv(path_events)

def _load_variables(
        path_variables
        ) -> pd.DataFrame:
    return pd.read_csv(path_variables)

__all__ = [
    # Primary entry points
    "load_raw_inputs",
    # Data classes
    "DynamicInputs",
    "DynamicResults"
]