# Load Scenarios


Load perturbations generate multiple load scenarios from an initial case file scenario. Each scenario consists of active and reactive power values for all loads in the network.




The module provides three main perturbation strategies:

### Comparison of Perturbation Strategies


<div class="center-table" markdown>

| Feature                      | `LoadScenariosFromAggProfile` | `Powergraph`                   | `PrecomputedProfile`          |
|-----------------------------|-------------------------------|-------------------------------|-------------------------------|
| **Global scaling**          | ✅ Yes                        | ✅ Yes                        | ❌ No                         |
| **Local (per-load) scaling**| ✅ Yes (via noise)            | ❌ No                         | ❌ No                         |
| **Reactive power perturbed**| ✅ Optional                   | ❌ No                         | ✅ From file                  |
| **Interpolation**           | ✅ Yes                        | ✅ Yes                        | ❌ No                         |
| **Use of real profile data**| ✅ Yes                        | ✅ Yes                        | ✅ From file                  |

</div>

## Mathematical Models

Let:

- $n$: Number of loads ($i \in \{1, \dots, n\}$)

- $K$: Number of scenarios ($k \in \{1, \dots, K\}$)

- $(p, q) \in (\mathbb{R}_{\geq 0}^n)^2$: Nominal active/reactive loads

- $\text{agg}^k$: Aggregated load profile value at time step $k$

- $u$: Maximum feasible global scaling factor (from OPF)

- $l = (1 - \text{global\textunderscore range}) \cdot u$: Minimum global scaling factor

- $\text{ref}^k = \text{MinMaxScale}(\text{agg}^k, [l, u])$: Scaled aggregate profile

- $\varepsilon_i^k \sim \mathcal{U}(1 - \sigma, 1 + \sigma)$: Active power noise

- $\eta_i^k \sim \mathcal{U}(1 - \sigma, 1 + \sigma)$: Reactive power noise (if enabled)



### `LoadScenariosFromAggProfile`

Generates load scenarios by scaling all loads of the grid using a global scaling factor derived from an aggregated load profile, while also applying local (load-level) noise to introduce heterogeneity across buses. Both active and reactive power can be perturbed.

The process includes:

1. Determining an upper bound $u$ for load scaling such that the network still
    supports a feasible optimal power flow (OPF) solution.
2. Setting the lower bound $l = (1 - \text{global\textunderscore range}) \cdot u$.
3. Min-max scaling the aggregate profile to the interval \([l, u]\).
4. Applying this global scaling factor to each load's nominal value with additive uniform noise.


For each load $i$ and scenario $k$:
$$
\tilde{p}_i^k = p_i \cdot \text{ref}^k \cdot \varepsilon_i^k
$$

$$
\tilde{q}_i^k =
\begin{cases}
q_i \cdot \text{ref}^k \cdot \eta_i^k & \text{if } \texttt{change\textunderscore reactive\textunderscore power} = \texttt{True} \\
q_i & \text{otherwise}
\end{cases}
$$

**Notes**

- The upper bound `u` is automatically determined by gradually increasing the base load (doing steps of size `step_size` and solving the OPF until it fails or reaches `max_scaling_factor`).

- The lower bound `l` is computed as a relative percentage (1-`global_range`) of `u`.

- Noise helps simulate local variability across loads within a global trend.

Sample config parameters:

```yaml
load:
  generator: "agg_load_profile" # Name of the load generator; options: agg_load_profile, powergraph, precomputed_profile
  agg_profile: "default" # Name of the aggregated load profile
  scenarios: 200 # Number of different load scenarios to generate
  sigma: 0.2 # max local noise
  change_reactive_power: true # If true, changes reactive power of loads. If False, keeps the ones from the case file
  global_range: 0.4 # Range of the global scaling factor. used to set the lower bound of the scaling factor
  max_scaling_factor: 4.0 # Max upper bound of the global scaling factor
  step_size: 0.025 # Step size when finding the upper bound of the global scaling factor
  start_scaling_factor: 0.8 # Initial value of the global scaling factor
```



### `Powergraph`
Generates load scenarios by scaling all loads of the grid with a normalized global scaling factor, derived from an aggregated load profile. Only the active power is perturbed; reactive power remains fixed across all scenarios. This follows the implementation of [PowerGraph: A power grid benchmark dataset for graph neural networks](https://arxiv.org/abs/2402.02827)


The reference profile is computed by normalizing an aggregated profile:

$$
\text{ref}^k = \frac{\text{agg}^k}{\max_k \text{agg}^k}
$$

Then, for each bus $i$ and scenario $k$:

$$
\tilde{p}_i^k = p_i \cdot \text{ref}^k
$$

and reactive power is kept constant:

$$
\tilde{q}_i^k = q_i
$$

Sample config parameters:

```yaml
load:
  generator: "powergraph"
  agg_profile: "default"           # Aggregated load profile name
  scenarios: 200                   # Number of load scenarios to generate
```

### `PrecomputedProfile`
Loads scenarios directly from a CSV or Excel file. Each row specifies the active and
reactive power for a given `(load_scenario, load)` pair. All pairs must be present.

Required columns:

- `load_scenario`: scenario index (0..n_scenarios-1)
- `load`: bus index (0..n_buses-1) using the network's continuous indexing
- `p_mw`: active power in MW
- `q_mvar`: reactive power in MVAr

Constraints:

- `load_scenario` and `load` must be integer-valued.
- All `(load_scenario, load)` pairs must be unique.
- The file must include exactly `n_buses * n_scenarios` rows.

Sample config parameters:

```yaml
load:
  generator: "precomputed_profile"
  scenario_file: "load-scenarios-precomputed.csv"
  scenarios: 200
```

Example files:

- `scripts/config/case14_config_precomputed_profile.yaml`
- `scripts/precomputed_load_profiles/load-scenarios-precomputed-case14.csv`

## Aggregated load profiles

The following load profiles are available in the `gridfm-datakit/load_profiles` directory:

- `default.csv`: Default load profile.

- [ERCOT load profiles](https://www.eia.gov/electricity/wholesalemarkets/data.php?rto=ercot):

    - `ercot_load_act_hr_2024_west.csv`: ERCOT load profile for the West region.
    - `ercot_load_act_hr_2024_south_central.csv`: ERCOT load profile for the South Central region.
    - `ercot_load_act_hr_2024_southern.csv`: ERCOT load profile for the Southern region.
    - `ercot_load_act_hr_2024_total.csv`: Total ERCOT load profile.
    - `ercot_load_act_hr_2024_far_west.csv`: ERCOT load profile for the Far West region.
    - `ercot_load_act_hr_2024_north.csv`: ERCOT load profile for the North region.
    - `ercot_load_act_hr_2024_north_central.csv`: ERCOT load profile for the North Central region.
    - `ercot_load_act_hr_2024_coast.csv`: ERCOT load profile for the Coast region.
    - `ercot_load_act_hr_2024_east.csv`: ERCOT load profile for the East region.
