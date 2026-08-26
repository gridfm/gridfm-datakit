# Dynamic Simulation

This module provides the time-domain (dynamic) data generation pipeline. See the
[Dynamic Simulation manual page](../manual/dynamic_simulation.md) for the
configuration and output reference.

## Entry point

### `generate_dynamic_data`

::: gridfm_datakit.dynamic.generate_dynamic.generate_dynamic_data

## Inputs

### `DynamicInputs`

::: gridfm_datakit.dynamic.DynamicInputs

### `DynamicResults`

::: gridfm_datakit.dynamic.DynamicResults

### `load_raw_inputs`

::: gridfm_datakit.dynamic.load_raw_inputs

## Processing

### `iter_dynamic_simulations`

::: gridfm_datakit.dynamic.process_dynamic.iter_dynamic_simulations

### `process_dynamic_simulations`

::: gridfm_datakit.dynamic.process_dynamic.process_dynamic_simulations

### `process_single_dynamic_simulation`

::: gridfm_datakit.dynamic.process_dynamic.process_single_dynamic_simulation

## Dynawo backend

### `DynawoMappings`

::: gridfm_datakit.dynamic.dynawo.DynawoMappings

### `generate_dynawo_mappings`

::: gridfm_datakit.dynamic.dynawo.generate_dynawo_mappings

### `get_dynawo_simulation_parameters`

::: gridfm_datakit.dynamic.dynawo.get_dynawo_simulation_parameters

### `get_dynawo_loadflow_parameters`

::: gridfm_datakit.dynamic.dynawo.get_dynawo_loadflow_parameters

### `compute_balanced_static_state_dynawo`

::: gridfm_datakit.dynamic.dynawo.simulate.compute_balanced_static_state_dynawo

### `run_dynawo_simulation`

::: gridfm_datakit.dynamic.dynawo.simulate.run_dynawo_simulation

## Availability checks

### `is_dynawo_available`

::: gridfm_datakit.dynamic.dynawo.api.is_dynawo_available

### `check_dynawo_available`

::: gridfm_datakit.dynamic.dynawo.api.check_dynawo_available

## Validation

### `validate_dynamic_data`

::: gridfm_datakit.validation.validate_dynamic_data
