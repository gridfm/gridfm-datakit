"""General module for dynamic simulations."""

from dataclasses import dataclass
import pandas as pd
from typing import Any

from gridfm_datakit.utils.param_handler import NestedNamespace

@dataclass
class DynamicInputs:
    """
    Solver-agnostic container for dynamic simulation inputs.

    Attributes are kept as pandas DataFrames to remain compatible
    with pypowsybl.dynamic's native input format and ease
    manipulation and inspection.

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
    Solver-agnostic container for dynamic simulation outputs.

    - dynamic_results: time-series data stored as a Zarr array or group,
      shaped (n_variables, n_timesteps) per scenario.
    - report: solver convergence / status report string.
    """
    dynamic_results: Any # zarr.Array or zarr.Group — (n_variables, n_timesteps)
    report: Any


def load_raw_inputs(
          args: NestedNamespace,
          input_format='csv'
          ) -> DynamicInputs:
    """Load dynamic inputs into a DynamicInputs object."""
    
    if input_format == 'csv':
        # TODO: recuperate the data from csv files.
        dynamic_models = [pd.read_csv(args.dynamic.input_files.dynamic_models_file), 
                          pd.read_csv(args.dynamic.input_files.automation_systems_file)
                          ]
        events = pd.read_csv(args.dynamic.input_files.events_file)
        variables = pd.read_csv(args.dynamic.input_files.variables_file)

    return DynamicInputs(
            dynamic_models=dynamic_models,
            events=events,
            variables=variables
    )

__all__ = [
    # Primary entry points
    "load_raw_inputs",
    # Data classes
    "DynamicInputs",
    "DynamicResults"
]