

# Power Flow Data Generation Script

This library generates power flow data with configurable parameters for load scenarios and topology perturbations.

![image](https://github.ibm.com/PowerGrid-FM/grid_data_synthetic/assets/474695/e6b81dfa-13ac-4e55-b7b8-b986aad6a268)


## Setup

### Create a Python Virtual Environment and Install Requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Install the Package in Editable Mode During Development

```bash
pip install -e .
```

### Usage

```bash
python dist_generate_pf_data.py --config path/to/config.yaml --data_path /path/to/data
```

### Command-Line Arguments

| Argument    | Type | Default                | Description                                                                                     |
|-------------|------|------------------------|-------------------------------------------------------------------------------------------------|
| `--config`  | str  | `config/default.yaml`   | (Required) Path to the configuration YAML file.                                                 |
| `--data_path` | str  | `../data`               | (Optional) Directory where to store the dataset. Defaults to the data folder one level up from the current working directory. |

### Example Commands

1. **Basic Data Generation Run**

   ```bash
   python dist_generate_pf_data.py --config config/default.yaml
   ```

2. **Custom Data Path**

   ```bash
   python dist_generate_pf_data.py --config config/default.yaml --data_path /dccstor/gridfm/data
   ```

### YAML Configuration Parameters

| **Section**              | **Parameter**                | **Type** | **Description**                                                                                     |
|--------------------------|------------------------------|----------|-----------------------------------------------------------------------------------------------------|
| **network**              | `name`                       | str      | Name of the power grid network.                                                                     |
|                          | `source`                     | str      | Data source for the grid; options: `pglib`, `pandapower`, `file`.                                  |
| **load**                 | `generator`                  | str      | Name of the load generator; options: `agg_load_profile`, `powergraph`.                             |
|                          | `agg_profile`                | str      | Name of the aggregated load profile; used when `generator` is `agg_load_profile`.                   |
|                          | `scenarios`                  | int      | Number of different load scenarios to generate.                                                     |
|                          | `sigma`                      | float    | Max local noise; used when `generator` is `agg_load_profile`.                                     |
|                          | `change_reactive_power`      | bool     | If true, changes reactive power of loads. If false, keeps the ones from the case file. Used when `generator` is `agg_load_profile`. |
|                          | `global_range`               | float    | Range of the global scaling factor. Used to set the lower bound of the scaling factor. Used when `generator` is `agg_load_profile`. |
|                          | `max_scaling_factor`         | float    | Max upper bound of the global scaling factor. Used when `generator` is `agg_load_profile`.        |
|                          | `step_size`                  | float    | Step size when finding the upper bound of the global scaling factor. Used when `generator` is `agg_load_profile`. |
|                          | `start_scaling_factor`       | float    | Initial value of the global scaling factor. Used when `generator` is `agg_load_profile`.          |
| **topology_perturbation**| `type`                       | str      | Type of topology generator; options: `n_minus_k`, `random`, `overloaded`, `none`.                  |
|                          | `k`                          | int      | Maximum number of components to drop in each perturbation; used when `type` is `n_minus_k` or `random`. |
|                          | `n_topology_variants`        | int      | Number of unique perturbed topologies per scenario; used when `type` is `n_minus_k`, `random`, or `overloaded`. |
| **settings**             | `num_processes`              | int      | Number of parallel processes to use.                                                               |
|                          | `data_dir`                   | str      | Directory to save generated data.                                                                  |
|                          | `large_chunk_size`           | int      | Number of load scenarios processed before saving.                                                  |
|                          | `no_stats`                   | bool     | If true, disables statistical calculations.                                                        |
|                          | `overwrite`                  | bool     | If true, overwrites existing files.                                                                |

**Disclaimer:** Not all parameters are relevant depending on the choice of `type` for topology perturbation and `generator` for load. Please refer to the specific configuration requirements for your use case.
