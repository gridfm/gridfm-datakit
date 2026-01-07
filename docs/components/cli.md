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
gridfm-datakit generate scripts/config/default.yaml
```

### Validate Data

Validate previously generated power flow data. Runs comprehensive validation checks for data integrity and physical consistency:

```bash
gridfm-datakit validate path/to/data/directory [--n-partitions N] [--sn-mva 100]
```

**Arguments:**
- `data_path`: Path to directory containing generated CSV files
- `--n-partitions N`: Number of partitions (of 200 scenarios) to sample for validation (default: 100). Use 0 to validate all partitions.
- `--sn-mva`: Base MVA used to scale power quantities (default: 100).

**Examples:**
```bash
# Validate with default sampling (100 partitions)
gridfm-datakit validate ./data_out/case24_ieee_rts/raw

# Validate custom number of partitions
gridfm-datakit validate ./data_out/case24_ieee_rts/raw --n-partitions 50

# Validate all partitions (slower but complete)
gridfm-datakit validate ./data_out/case24_ieee_rts/raw --n-partitions 0
```

The validation command performs the following checks:

#### Y-Bus Consistency
- Consistency of bus admittance matrix with branch admittance data
- Y-bus matrix structure validation

#### Branch Constraints
- Deactivated lines have zero power flows and admittances
- Computed vs stored power flow consistency
- Branch loading limits (OPF mode only)

#### Generator Constraints
- Deactivated generators have zero power output
- Generator power limits validation
- Reactive power limits (OPF mode only)

#### Power Balance
- Bus generation consistency between bus_data and gen_data
- Power Balance

#### Data Integrity
- Scenario indexing consistency across all files
- Bus indexing consistency
- Data completeness and missing value checks


### Stats

Compute and display statistics from generated power flow data:

```bash
gridfm-datakit stats path/to/data/directory [--sn-mva 100]
```

**Arguments:**
- `data_path`: Path to directory containing generated parquet files (`bus_data.parquet`, `branch_data.parquet`, `gen_data.parquet`)
- `--sn-mva`: Base MVA used to scale power quantities (default: 100).

**Example:**
```bash
gridfm-datakit stats ./data_out/case24_ieee_rts/raw
```

This command:
1. Computes aggregated statistics across sampled partitions:
   - Number of active generators and branches per scenario
   - Branch loading metrics (overloads, maximum loading, all branch loadings)
   - Power balance errors (active and reactive, normalized by number of buses)
2. Generates and saves `stats_plot.png` containing histogram distributions of these metrics

The statistics help assess dataset quality, identify constraint violations (overloads), and verify power balance consistency. See the [stats module](../components/stats.md) documentation for details.

### Plots

Generate violin plots for all bus features and save them to disk:

```bash
gridfm-datakit plots path/to/data/directory [--output-dir DIR] [--sn-mva 100]
```

**Arguments:**
- `data_path`: Path containing `bus_data.parquet`
- `--output-dir DIR` (optional): Directory where plots are saved (default: `data_path/feature_plots`)
- `--sn-mva` (optional): Base MVA used to normalize Pd/Qd/Pg/Qg (default: 100)

**Example:**
```bash
gridfm-datakit plots ./data_out/case24_ieee_rts/raw --sn-mva 100
```

This command reads `bus_data.parquet`, normalizes power columns by `sn_mva`, and writes violin plots named `distribution_{feature_name}.png` to the output directory for quick visualization of feature distributions.
