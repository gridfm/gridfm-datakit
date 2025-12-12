# PF Delta Batch Converter

Converts power flow solution JSON files (PowerModels/PGLib format) into parquet outputs structured for power flow analysis:
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
python batch_convert_pfdelta.py --data-dir /path/to/raw/ --out-dir /path/to/converted/
```

### Arguments

- `--data-dir`: Directory containing `sample_*.json` files (non-recursive)
  - Default: `/Users/apu/to_del/gridfm-datakit/pfdelta/data/case14_ieee_n_minus_one/raw`
- `--out-dir`: Directory for aggregated parquet outputs
  - Default: `/Users/apu/to_del/gridfm-datakit/pfdelta/data/case14_ieee_n_minus_one/converted`

### Optional Arguments

- `--chunk-size`: Files per processing chunk (default: 2000)

## Key Features

### Scenario Indexing
- Filenames: `sample_N.json` → scenario index = N-1 (zero-based)
- Example: `sample_1.json` = scenario 0, `sample_1000.json` = scenario 999

### Unified Admittance Calculation
- Uses `compute_branch_admittances()` for consistent Y-matrix computation
- Works for both AC lines (tap=1.0, shift=0.0) and transformers
- Splits shunt susceptance equally between bus endpoints (b/2 at each end)

### Power Scaling
- All powers scaled by `baseMVA` (MW/MVAR if baseMVA in MVA)
- Voltage angles stored in degrees (converted from radians)

### Validation
- Verified via `validate_admittance_calculations()` in validation module
- Checks that stored Y-matrix values match calculated values from branch parameters
- Tolerance: 1e-8

## Data Structure

### Expected Input Format

PowerModels/PGLib JSON structure:
```json
{
  "network": {
    "baseMVA": 100.0,
    "per_unit": true,
    "bus": {"1": {...}, "2": {...}, ...},
    "gen": {"1": {...}, "2": {...}, ...},
    "load": {"1": {...}, "2": {...}, ...},
    "shunt": {"1": {...}, "2": {...}, ...},
    "branch": {"1": {...}, "2": {...}, ...}
  },
  "solution": {
    "solution": {
      "baseMVA": 100.0,
      "bus": {"1": {"vm": 1.0, "va": 0.0}, ...},
      "gen": {"1": {"pg": 50.0, "qg": 0.0}, ...},
      "branch": {"1": {"pf": 0.0, "qf": 0.0, "pt": 0.0, "qt": 0.0}, ...}
    },
    "objective": 12345.67
  }
}
```

### Output Files

All files are saved as **scenario-partitioned parquets** with partition key `scenario_partition`:
- Each partition contains 100 consecutive scenarios
- Enables efficient parallel reads and reduces memory overhead

### Key Columns

**`branch_data.parquet`**: Branch/line/transformer parameters and power flows
- `scenario` (int): Scenario index (0-based)
- `idx` (int): Branch index within scenario
- `from_bus`, `to_bus` (int): Bus indices (0-based)
- `pf`, `qf`, `pt`, `qt` (float): Power flows in MW/MVAR
- `Yff_r/i`, `Yft_r/i`, `Ytf_r/i`, `Ytt_r/i` (float): Y-matrix admittance components
- `tap`, `shift` (float): Transformer tap ratio and phase shift (degrees)
- `ang_min`, `ang_max` (float): Angle difference limits (degrees)
- `rate_a` (float): MVA rating
- `r`, `x`, `b` (float): Impedance and shunt susceptance
- `br_status` (float): 1.0 = active, 0.0 = deactivated

**`bus_data.parquet`**: Bus voltage and power data
- `scenario`, `bus` (int): Scenario and bus indices
- `Pd`, `Qd` (float): Demand in MW/MVAR
- `Pg`, `Qg` (float): Generation in MW/MVAR
- `Vm`, `Va` (float): Voltage magnitude (p.u.) and angle (degrees)
- `PQ`, `PV`, `REF` (int): Bus type indicators (1, 0, 0 for PQ bus, etc.)
- `vn_kv` (float): Nominal voltage (kV)
- `min_vm_pu`, `max_vm_pu` (float): Voltage magnitude limits (p.u.)
- `GS`, `BS` (float): Shunt conductance/susceptance (p.u.)

**`gen_data.parquet`**: Generator outputs and costs
- `scenario`, `idx` (int): Scenario and generator indices
- `bus` (int): Bus where generator is connected
- `p_mw`, `q_mvar` (float): Power output in MW/MVAR
- `min_p_mw`, `max_p_mw` (float): Active power limits
- `min_q_mvar`, `max_q_mvar` (float): Reactive power limits
- `cp0_eur`, `cp1_eur_per_mw`, `cp2_eur_per_mw2` (float): Cost coefficients (c₀, c₁, c₂)
- `in_service` (int): 1 if active, 0 if offline
- `is_slack_gen` (int): 1 if at slack bus, 0 otherwise

**`y_bus_data.parquet`**: Admittance matrix entries
- `scenario` (int): Scenario index
- `index1`, `index2` (int): Y-matrix row/column indices
- `G`, `B` (float): Conductance and susceptance

## Assumptions and Assertions

### Physical Network Structure
- Buses numbered consecutively 1 to N
- Generators numbered consecutively 1 to NG
- Branches numbered consecutively 1 to NB
- Bus keys in JSON match bus_i field
- Generator keys in JSON match index field
- Branch keys in JSON match index field

### Branch Parameters
- `br_x != 0.0` for all branches (no zero reactance)
- `br_status ∈ {0, 1}` (deactivated lines have zero power flows and admittances)
- `g_fr == 0.0` and `g_to == 0.0` for all branches (no series conductance)
- `b_fr == b_to` for all branches (symmetric shunt susceptance)
- `tap > 0.0` for all transformers; tap = 1.0 for AC lines
- For AC lines: shunt susceptance at each end

### Bus and Generator Indices
- All bus indices 1-based and within [1, n_buses]
- Generator `gen_bus` references valid bus index
- Load `load_bus` references valid bus index
- Shunt `shunt_bus` references valid bus index
- Bus types: 1 (PQ), 2 (PV), 3 (slack), 4 (inactive/rejected)

### Solution Validation
- Generator cost: ∑(c₀ + c₁·pₘ + c₂·pₘ²) must match solution objective within 1e-3
- Admittance matrices computed from branch parameters must match stored Y-matrix within 1e-8 tolerance
- Slack bus must be exactly one (type 3)
- Deactivated generators (gen_status = 0) contribute zero to generation
