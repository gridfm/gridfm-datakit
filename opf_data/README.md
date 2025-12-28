# OPF Data Batch Converter

Converts OPF (Optimal Power Flow) JSON files into parquet outputs structured for power flow analysis:
- `branch_data.parquet` - Branch/line/transformer data with admittance matrices
- `bus_data.parquet` - Bus voltage and power injection data
- `gen_data.parquet` - Generator outputs and costs
- `y_bus_data.parquet` - Admittance matrix for power flow calculations

## Processing Pipeline

The converter processes files in **fixed-size chunks** with **multiprocessing** and **appends sorted results incrementally** to parquet files. This ensures:
- ✅ Consistent global ordering (monotone scenario indices)
- ✅ Efficient memory usage (doesn't load all files at once)
- ✅ Scenario-based partitioning for easy data access

### Example usage

```bash
python batch_convert.py /path/to/opf_data/ /path/to/output/
```

### Arguments

- `data_dir` (positional): Directory containing `example*.json` files (searched recursively)
- `out_dir` (positional): Directory for aggregated parquet outputs

### Optional Arguments

- `--chunk-size`: Files per processing chunk (default: 2000)

## Key Features

### Unified Admittance Calculation
- Uses `compute_branch_admittances()` for consistent Y-matrix computation
- Works for both AC lines (tap=1.0, shift=0.0) and transformers
- Splits shunt susceptance equally between bus endpoints

### Validation
- Verified via `validate_admittance_calculations()` in validation module
- Checks that stored Y-matrix values match calculated values from branch parameters
- Tolerance: 1e-8

## Data Structure

### Output Files

All files are saved as **scenario-partitioned parquets** with partition key `scenario_partition`:
- Each partition contains 100 consecutive scenarios
- Enables efficient parallel reads and reduces memory overhead

### Key Columns

**`branch_data.parquet`**: Branch/line/transformer parameters and power flows
- `from_bus`, `to_bus` (int): Bus indices
- `pf`, `qf`, `pt`, `qt` (float): Power flows in MW/MVAR
- `Yff_r/i`, `Yft_r/i`, `Ytf_r/i`, `Ytt_r/i` (float): Y-matrix admittance components
- `tap`, `shift` (float): Transformer tap and phase shift
- `r`, `x`, `b` (float): Impedance and shunt susceptance
- `br_status` (float): 1.0 = active, 0.0 = deactivated

**`bus_data.parquet`**: Bus voltage and power data
- `Pd`, `Qd` (float): Demand in MW/MVAR
- `Pg`, `Qg` (float): Generation in MW/MVAR
- `Vm`, `Va` (float): Voltage magnitude (p.u.) and angle (degrees)
- `PQ`, `PV`, `REF` (int): Bus type indicators

**`gen_data.parquet`**: Generator outputs and costs
- `p_mw`, `q_mvar` (float): Power output
- `min_p_mw`, `max_p_mw` (float): Power limits
- `cp0_eur`, `cp1_eur_per_mw`, `cp2_eur_per_mw2` (float): Cost coefficients
- `in_service` (int): 1 if active, 0 if offline

**`y_bus_data.parquet`**: Admittance matrix entries
- `index1`, `index2` (int): Y-matrix row/column indices
- `G`, `B` (float): Conductance and susceptance

## Assumptions and Assertions

### Physical Network Structure
- `br_x != 0.0` for all branches (no zero reactance)
- `tap_mag > 0.0` for all branches (AC lines: tap=1.0)
- `b_fr == b_to` for all branches (symmetric shunt)
- Bus type ∈ {1, 2, 3}; type 4 (inactive) rejected
- All index references within bounds

### Solution Validation
- Generator cost: ∑(c₀ + c₁·pₘ + c₂·pₘ²) must match objective within 1e-3
- Admittance matrices must match stored Y-matrix within 1e-8 tolerance
