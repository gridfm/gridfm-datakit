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

### Stats

Compute and display statistics from generated power flow data:

```bash
gridfm-datakit stats path/to/data/directory
```

**Arguments:**
- `data_path`: Path to directory containing generated parquet files (`bus_data.parquet`, `branch_data.parquet`, `gen_data.parquet`)

**Example:**
```bash
gridfm-datakit stats ./data_out/case24_ieee_rts/raw
```

This command:
1. Computes aggregated statistics across all scenarios:
   - Number of active generators and branches per scenario
   - Branch loading metrics (overloads, maximum loading, all branch loadings)
   - Power balance errors (active and reactive, normalized by number of buses)
2. Generates and saves `stats_plot.png` containing histogram distributions of these metrics

The statistics help assess dataset quality, identify constraint violations (overloads), and verify power balance consistency. See the [stats module](../components/stats.md) documentation for details.

### Feature Plots

Plot distributions for all bus features across buses and save PNGs:

```bash
gridfm_datakit plots path/to/data/directory [--output-dir DIR] [--sn-mva 100]
```

**Arguments:**
- `data_path`: Path containing `bus_data.parquet`
- `--output-dir DIR` (optional): Where to save plots (default: `data_path/feature_plots`)
- `--sn-mva` (optional): Base MVA to normalize Pd/Qd/Pg/Qg (default: 100)

**Examples:**
```bash
# Generate feature plots into the default directory (data_path/feature_plots)
gridfm_datakit plots ./data_out/case24_ieee_rts/raw

## Validation Checks

The validation command performs the following checks:

### Y-Bus Consistency
- Consistancy of bus admittance matrix with branch admittance data
- Y-bus matrix structure validation

### Branch Constraints
- Deactivated lines have zero power flows and admittances
- Computed vs stored power flow consistency
- Branch loading limits (OPF mode only)

### Generator Constraints
- Deactivated generators have zero power output
- Generator power limits validation
- Reactive power limits (OPF mode only)

### Power Balance
- Bus generation consistency between bus_data and gen_data
- Power Balance

### Data Integrity
- Scenario indexing consistency across all files
- Bus indexing consistency
- Data completeness and missing value checks

### `main`

::: gridfm_datakit.cli.main
