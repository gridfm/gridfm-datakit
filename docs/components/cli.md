# CLI

This module provides a command-line interface for generating and validating power flow data.

## Commands

### Generate Data

Generate power flow data from a configuration file:

```bash
gridfm-datakit generate path/to/config.yaml
```

**Arguments:**
- `config`: Path to the YAML configuration file

**Example:**
```bash
gridfm-datakit generate scripts/config/case24_ieee_rts.yaml
```

### Validate Data

Validate previously generated power flow data. Runs comprehensive validation checks for data integrity and physical consistency:

```bash
gridfm-datakit validate path/to/data/directory [--n-scenarios N]
```

**Arguments:**
- `data_path`: Path to directory containing generated CSV files
- `--n-scenarios N`: Number of scenarios to sample for validation (default: 100). Use 0 to validate all scenarios.

**Examples:**
```bash
# Validate with default sampling (100 scenarios)
gridfm-datakit validate ./data_out/case24_ieee_rts/raw

# Validate with custom scenario sampling
gridfm-datakit validate ./data_out/case24_ieee_rts/raw --n-scenarios 50

# Validate all scenarios (slower but complete)
gridfm-datakit validate ./data_out/case24_ieee_rts/raw --n-scenarios 0
```

## Validation Checks

The validation command performs the following checks:

### Y-Bus Consistency
- Consistancy of bus admittance matrix with branch admittance data
- Y-bus matrix structure validation

### Branch Constraints
- Deactivated lines have zero power flows and admittances
- Computed vs stored power flow consistency
- Branch loading limits (secure mode only)

### Generator Constraints
- Deactivated generators have zero power output
- Generator power limits validation
- Reactive power limits (secure mode only)

### Power Balance
- Bus generation consistency between bus_data and gen_data
- Power Balance

### Data Integrity
- Scenario indexing consistency across all files
- Bus indexing consistency
- Data completeness and missing value checks

### `main`

::: gridfm_datakit.cli.main
