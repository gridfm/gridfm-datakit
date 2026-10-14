# Dynamic Simulation

The dynamic pipeline extends the static one with a time-domain (RMS/phasor)
simulation step. Instead of a single power flow snapshot per scenario, it
produces a **trajectory**: the response of the grid to a disturbance, sampled
over the simulation window.

The time-domain solver is [Dynaωo](https://dynawo.github.io/), driven through
`pypowsybl.dynamic`. It is currently the only backend
(`dynamic.dynamic_solver: dynawo`).

## What a run does

For every load scenario, and for every topology perturbation of it:

1. **Load scenario** is applied to the network (same generators as the static
   pipeline), followed by the generation and admittance perturbations.
2. **OPF** is solved with PowerModels.jl → an optimal dispatch.
3. **Set-points are pushed to pypowsybl**, then an **AC power flow** is solved
   with OpenLoadFlow. This is the *balanced initial state*: Dynawo initialises
   the grid state from it.
4. **Dynawo runs the time-domain simulation** once per event variant, each on
   its own copy of that balanced network, applying the events of `events_file`
   or, with [`dynamic.event_perturbation`](#dynamicevent_perturbation-optional),
   the events drawn for that variant, and recording the monitored variables.
5. The static snapshot (Parquet) and the trajectory (Zarr) are written, both
   tagged with the same `(scenario_index, perturbation_index, event_index)` key.

Each `(scenario, perturbation, event)` triple is one **sample**. `event_index`
numbers the event variants of a topology variant: always 0 with `events_file`,
0 to `n_event_variants - 1` with random events. Load scenarios are
distributed across worker processes (the perturbations of a single scenario run
sequentially inside one worker), and results are written incrementally, one large
chunk at a time, so peak memory tracks `settings.large_chunk_size` rather than
the size of the whole dataset.

!!! warning "The dynamic model set is fixed across samples"
    The dynamic models and automation systems come from the CSV input tables and
    are **identical for every sample**. So are the events, unless
    `dynamic.event_perturbation` draws them. What varies from sample to
    sample is the operating point Dynawo starts from, plus, with
    `topology_perturbation` enabled, which elements are in service and hence
    which of those dynamic models Dynawo instantiates.

!!! warning "`generation_perturbation` does not work here"
    Step 1 accepts the block, but it cannot do its job. Dynamic runs require
    `reader: powsybl`, and pypowsybl carries no generator cost data in any format
    it reads, so every generator is loaded with the same placeholder cost
    (`c2=0, c1=1, c0=0`). See [Perturbations](#perturbations).

## Prerequisites

Two independent things are required, and each fails differently.

### 1. The Python extra

```bash
pip install 'gridfm-datakit[dynamic]'
```

This pulls in `pypowsybl` (the `pypowsybl.dynamic` API) and `zarr` (the
time-series store). Julia/PowerModels is still needed for the OPF step; see
[Installation](../installation.md).

### 2. A local Dynawo installation

Dynawo is a native solver and is **not bundled with pypowsybl**:
`import pypowsybl.dynamic` succeeds without it. Download a release from
[dynawo.github.io/install](https://dynawo.github.io/install/) and extract it,
e.g. to `~/dynawo`:

```bash
DYNAWO_VERSION=1.7.0
curl -fsSL -o /tmp/dynawo.zip \
  "https://github.com/dynawo/dynawo/releases/download/v${DYNAWO_VERSION}/Dynawo_Linux_v${DYNAWO_VERSION}.zip"
unzip -q /tmp/dynawo.zip -d "$HOME"
```

Then declare it to powsybl in `~/.itools/config.yml`:

```yaml
dynawo:
  homeDir: /home/<user>/dynawo
  debug: false
```

Set `POWSYBL_CONFIG_DIR` (and optionally `POWSYBL_CONFIG_NAME`) to use a
different location; `.yml`, `.yaml` and `.xml` config files are all recognised.

### Checking the installation

```bash
~/dynawo/dynawo.sh version
python -c "from gridfm_datakit.dynamic.dynawo.api import check_dynawo_available; check_dynawo_available()"
```

The same check runs automatically at the start of every dynamic run, before any
worker is spawned, so a missing installation surfaces as an actionable message
instead of an opaque `DynawoSimulationProvider could not be instantiated` from
deep inside the solver.

## Network requirements

Dynamic simulations require the **powsybl reader**:

```yaml
network:
  name: IEEE14
  reader: powsybl        # required, the run is rejected otherwise
  source: file
  file: grids/IEEE14.iidm
```

The dynamic model mappings are keyed on the network's **element IDs**, so the
network file must be one whose IDs you can reference from the input tables, in
practice an XIIDM/IIDM or CGMES file. Loading a MATPOWER `.m` case works, but
its element IDs are generated during conversion and are not stable to write
mappings against.

!!! danger "Use the IDs from the network file, not from `get_*()`"
    `pypowsybl.network.get_generators()` and friends can return slightly
    contaminated IDs (typically with an extra `_`). Writing those into the input
    tables produces a dynamic model Dynawo cannot instantiate. Read the IDs from
    the network file itself.

## The four input tables

The dynamic behaviour is described by four CSV (or Parquet) files, declared
under `dynamic.input_files`. CSV delimiters are sniffed, so `,`, `;` and tab are
all accepted. The required columns of all four tables are validated up front.

Values are checked only where a typo would otherwise be swallowed: the
`category_name` of `automation_systems`, the `event_name` of `events` and the
`type` of `variables` are matched against the accepted sets, and a bad one is
reported alongside them. Everything else (`model_name`, `static_id`, `model_id`
and the `category_name` of `static_element_dynamic_models`) is handed to Dynawo
as written and can only fail there, as a failed model instantiation.

### `static_element_dynamic_models_file`

One row per network element to equip with a dynamic model. Elements not listed
here are given Dynawo's default model.

| Column | Description |
| --- | --- |
| `category_name` | Category of the dynamic model, e.g. `SynchronousGenerator`, `LoadOneTransformerTapChanger` |
| `static_id` | Element ID in the network file |
| `parameter_set_id` | `set id` of a parameter group in the `.par` file |
| `model_name` | Dynawo model name, e.g. `GeneratorSynchronousFourWindingsProportionalRegulations` |

```csv
category_name,static_id,parameter_set_id,model_name
SynchronousGenerator,_GEN____1_SM,Generator1,GeneratorSynchronousFourWindingsProportionalRegulations
SynchronousGenerator,_GEN____6_SM,Generator6,GeneratorSynchronousThreeWindingsProportionalRegulations
LoadOneTransformerTapChanger,_LOAD___6_EC,GenericLoadOneTransfo,LoadOneTransformerTapChanger
```

### `automation_systems_file`

One row per automation system. These are not attached to a static element, so
they carry their own identifier and a free-form parameter string.

| Column | Description |
| --- | --- |
| `category_name` | Automation-system category (see table below) |
| `dynamic_model_id` | Identifier you give the automation system; usable as a `model_id` in `variables` |
| `parameter_set_id` | `set id` in the `.par` file |
| `params` | `key1=value1;key2=value2;…`, with keys that depend on the category |
| `model_name` | Dynawo model name, e.g. `UnderVoltage` |

Accepted `category_name` values and the `params` keys each one expects:

| `category_name` | `params` keys |
| --- | --- |
| `OverloadManagementSystem` | `controlled_branch`, `i_measurement`, `i_measurement_side` |
| `TwoLevelOverloadManagementSystem` | `controlled_branch`, `i_measurement_1`, `i_measurement_1_side`, `i_measurement_2`, `i_measurement_2_side` |
| `PhaseShifterBlockingI` | `phase_shifter_id` |
| `PhaseShifterI` | `transformer` |
| `PhaseShifterP` | `transformer` |
| `TapChanger` | `static_id`, `side` |
| `TapChangerBlocking` | `rfo_df`, `mp1_df`, `mp2_df`, `mp3_df`, `mp4_df`, `mp5_df` (**not usable, see below**) |
| `UnderVoltageAutomationSystem` | `generator` |

!!! danger "`TapChangerBlocking` cannot be configured from CSV"
    Its six parameters are DataFrames (hence the `_df` suffix), and the `params`
    column only expresses flat `key=value` scalars. There is no convention yet
    for describing a DataFrame in it.

    The category is still *accepted* by the input validation, so a row using it
    passes the up-front checks and then fails further down. Supporting it means
    first fixing a serialisation convention for those columns.

```csv
category_name,dynamic_model_id,parameter_set_id,params,model_name
UnderVoltageAutomationSystem,UVA,UnderVoltageAutomatonGenerator3,generator=_GEN____3_SM;,UnderVoltage
```

A run with no automation system still needs the file. Write the header row only.

### `events_file`

The disturbance sequence. This is what makes the trajectory non-trivial.

Read only when `dynamic.event_perturbation` is absent or of type `file`. With
type `none` or `random` the key is optional, and ignored with a warning when set.

| Column | Description |
| --- | --- |
| `event_name` | Event type (see table below) |
| `static_id` | ID of the element the event applies to |
| `start_time` | Event time, in seconds of simulation time |
| `params` | `key1=value1;…`, with keys that depend on the event type |

| `event_name` | `params` keys |
| --- | --- |
| `ActivePowerVariation` | `delta_p`: for a load modelled as `LoadAlphaBeta`, `LoadZIP` or `ElectronicLoad`, a fraction of its reference active power (0.1 raises it by 10 %); for a generator, per unit of its nominal power `PNom` |
| `ReactivePowerVariation` | `delta_q`: for a load modelled as `LoadAlphaBeta`, `LoadZIP` or `ElectronicLoad`, a fraction of its reference reactive power |
| `ReferenceVoltageVariation` | `delta_u`: per unit of the generator's nominal voltage `UNom` |
| `NodeFault` | `fault_time`, `r_pu`, `x_pu` |
| `Disconnect` | `disconnect_only` (optional; leave the value empty to disconnect the whole element) |

```csv
event_name,static_id,start_time,params
Disconnect,_GEN____2_SM,50,disconnect_only=;
```

`start_time` must lie inside the `[start_time, stop_time]` window of
`dynamic.solver_parameters`, otherwise the event never fires. These times are
not checked.

### `variables_file`

What to record. One row per monitored variable; repeat `model_id` to monitor
several variables of the same element.

| Column | Description |
| --- | --- |
| `type` | `Curve` (full time series) or `FinalStateValue` (end-of-simulation scalar) |
| `model_id` | ID of the monitored element, or the `dynamic_model_id` of an automation system |
| `variables` | One variable name per row; see below for where the available names are listed |

The variables a model exposes are listed in its description file, shipped with
Dynawo under `<dynawo_home>/ddb/`. That file is named after the **Dynawo model
library**, which is usually, but not always, the `model_name` written in the
input tables: pypowsybl's `UnderVoltage` is Dynawo's `UnderVoltageAutomaton`, so
its variables are in `ddb/UnderVoltageAutomaton.desc.xml`. The prefix of the
variable names is the reliable hint (`underVoltageAutomaton_…`).

```csv
type,model_id,variables
Curve,_BUS____2_TN,U_value
Curve,UVA,underVoltageAutomaton_UMinPu
Curve,_GEN____1_SM,generator_efdPu_value
FinalStateValue,_GEN____1_SM,generator_UPu
```

!!! note "At least one `Curve` row is required"
    The time-series store is built from `Curve` rows only. A table with none is
    rejected up front. Without that check the simulation runs, monitors nothing,
    and fails much later in the writer, after all the compute.

    `FinalStateValue` rows are optional. When present they are written to a
    separate per-sample scalar table, not to the Zarr store.

## The Dynawo parameter file (`.par`)

Every `parameter_set_id` in the input tables refers to a `<set id="...">` block
in a Dynawo `.par` file, an XML file holding the physical parameters of the
models (machine constants, regulator gains, tap-changer settings), the network
parameters, and the solver settings. It is referenced from
`dynamic.solver_parameters`, and Dynawo distributions ship examples under
`<dynawo_home>/examples/`.

To find out which parameters a given model requires, read the description file
Dynawo ships for it under `<dynawo_home>/ddb/`, one per model library, listing
its parameters and the variables it exposes. For instance, the set `Generator1`
referenced above must supply what
`ddb/GeneratorSynchronousFourWindingsProportionalRegulations.desc.xml` declares.

`scripts/dynamic_example/grids/IEEE14.par` is a working example. Its `set id`s
are `Network`, `SimplifiedSolver`, `Generator1`, `Generator2`, `Generator3`,
`Generator6`, `Generator8`, `GenericLoadOneTransfo`, `GenericLoadTwoTransfos`,
`OmegaRef`, `GeneratorDisconnection` and `UnderVoltageAutomatonGenerator3`. The
example input tables reference a subset of those; `Network` and
`SimplifiedSolver` are reached through `dynamic.solver_parameters` instead, and a
`.par` may carry sets that a given run never uses.

## Configuration

### `dynamic.solver_parameters`

`start_time` and `stop_time` (both in seconds) are **required**. Every other key
is optional and is passed through to the Dynawo provider:

| Config key | Dynawo provider parameter |
| --- | --- |
| `parameters_file` | `parametersFile` |
| `network_parameters_file` | `network.parametersFile` |
| `network_parameters_id` | `network.parametersId` |
| `solver_type` | `solver.type` (`SIM` for the fixed-step simplified solver, `IDA` for the variable-step one) |
| `solver_parameters_file` | `solver.parametersFile` |
| `solver_parameters_id` | `solver.parametersId` |
| `precision` | `precision` |

A missing required key or an unsupported one is rejected, naming the accepted
set. Values of `none` or `""` are dropped rather than forwarded.

!!! note "These parameters are validated inside the workers"
    Unlike the Dynawo availability check, `dynamic.solver_parameters` and
    `dynamic.loadflow_parameters` are only built once a worker starts. A bad key
    therefore fails every chunk: the parent logs
    `Error in dynamic chunk: dynamic.solver_parameters: unsupported key(s) …`
    and the run ends with `Dynamic generation produced no samples`. The first
    message is the one that names the offending key.

### `dynamic.loadflow_parameters` (optional)

Governs the AC power flow that produces the balanced initial state. The defaults
match the static pipeline's `get_default_lf_params()`, but are pinned here so the
dynamic pipeline keeps them whatever the static default becomes:

```yaml
dynamic:
  loadflow_parameters:
    distributed_slack: false
    read_slack_bus: true
    write_slack_bus: true
    provider_parameters:
      slackBusSelectionMode: LARGEST_GENERATOR
```

Dynawo initialises each synchronous machine from this solution, so the slack has
to sit on a machine that carries a dynamic model. OpenLoadFlow's own default
(`MOST_MESHED`) can select a bus with no generator at all, leaving Dynawo to
initialise from a state its machine models cannot reproduce. `provider_parameters`
is a free-form pass-through to OpenLoadFlow; the other three keys are the only
top-level ones accepted.

### `dynamic.event_perturbation` (optional)

Draws the events of each sample instead of replaying `events_file`:

```yaml
dynamic:
  event_perturbation:
    type: random
    n_event_variants: 3
    scenarios:
      - name: trip_then_fault
        anchor: random
        weight: 3
        start_time: {distribution: uniform, low: 20, high: 80}
        events:
          - type: Disconnect
            target: {element: generator, distance: [0, 2]}
          - type: NodeFault
            target: {element: bus, distance: [0, 1]}
            delay: {distribution: uniform, low: 0.1, high: 0.3}
            params: {fault_time: 0.1, r_pu: 0, x_pu: 0.05}
      - name: load_step
        weight: 1
        start_time: {distribution: uniform, low: 20, high: 80}
        events:
          - type: ActivePowerVariation
            target: {element: load, distance: 0}
            params: {delta_p: {distribution: uniform, low: -0.2, high: 0.2}}
```

Each event variant draws `trip_then_fault` three times out of four: a generator
trip, then a fault 0.1 to 0.3 s later on a bus next to the anchor. Otherwise it
draws a load step.

| `type` | Events of every sample |
| --- | --- |
| `none` | None. `events_file` is optional and ignored |
| `file` | The rows of `events_file`. The default when the block is absent |
| `random` | Drawn per event variant from `scenarios` |

| Key | Default | Meaning |
| --- | --- | --- |
| `type` | required | `none`, `file` or `random` |
| `n_event_variants` | `1` | Event variants per topology variant; must be 1 unless `random` |
| `scenarios` | | `random` only, one or more entries, one drawn per event variant by `weight` |
| `scenarios[].name` | required | Unique, recorded in the `scenario` column of `events.parquet` and named in `raw/error.log` |
| `scenarios[].weight` | `1` | Relative chance of drawing the scenario, a positive number |
| `scenarios[].anchor` | `random` | The bus distance targets are placed around: a bus ID of the network file, or `random` for one drawn per event variant |
| `scenarios[].start_time` | required | Event time in seconds, a value spec, positive |
| `scenarios[].events` | required | One or more entries, their targets distinct within the scenario |
| `events[].type` | required | `Disconnect` (the whole element), `NodeFault`, `ActivePowerVariation`, `ReactivePowerVariation` or `ReferenceVoltageVariation` |
| `events[].target.element` | required unless `static_id` | `bus`, `generator`, `load`, `line` or `transformer`, narrowed by the type below |
| `events[].target.distance` | required unless `static_id` | Hops from the anchor, an integer or an inclusive `[min, max]` |
| `events[].target.static_id` | | In place of `element` and `distance`, the ID of a bus, generator, load, line or transformer in service in the network file, checked before any simulation. A sample whose topology variant took it out of service fails. The distance targets of the scenario never pick it |
| `events[].params` | | Every param of the type, each a value spec. Required for every type but `Disconnect`, which takes none |
| `events[].delay` | `0` | Seconds after the scenario's `start_time`, a value spec, non-negative. The event time is `start_time + delay` |

| `type` | `target.element` | `params` |
| --- | --- | --- |
| `NodeFault` | `bus` | `fault_time` positive, `r_pu` and `x_pu` non-negative |
| `ActivePowerVariation` | `generator`, `load` | `delta_p`, any sign, unit as in `events_file` |
| `ReactivePowerVariation` | `generator`, `load` | `delta_q`, any sign, unit as in `events_file` |
| `ReferenceVoltageVariation` | `generator` | `delta_u`, any sign, unit as in `events_file` |

A node fault on a bus drawn at `distance` 0 or 1 hop from a random anchor:

```yaml
        events:
          - type: NodeFault
            target: {element: bus, distance: [0, 1]}
            params:
              fault_time: {distribution: normal, mean: 0.1, std: 0.02, min: 0.05, max: 0.2}
              r_pu: 0
              x_pu: {distribution: uniform, low: 0.01, high: 0.1}
```

Every level rejects unknown keys and names their path, for example
`dynamic.event_perturbation.scenarios[0].probability`.

A value spec is a number, fixed, or one of:

| Form | Draw |
| --- | --- |
| `{distribution: normal, mean, std, min, max}` | Normal truncated to `[min, max]`, both optional |
| `{distribution: uniform, low, high}` | Uniform on `[low, high)` |
| `{distribution: choice, values, weights}` | One of `values`, `weights` optional |

Values outside the bounds are redrawn, never clipped.

`distance` counts branch hops from the anchor over the lines, two-winding
transformers and three-winding transformers in service after the topology
perturbation, a three-winding transformer joining each pair of its connected
ends. A generator or load sits at its bus's distance, a branch at its nearer
end's. A three-winding transformer is never a target. Each target is drawn among
the elements that leave every later target of the scenario an element of its
own. A random anchor is redrawn, up to 100 times, when its targets cannot all be
placed around it. A fixed
anchor is checked against the network file before any simulation, and so is
every `static_id`: it must be in service and of an element type its event
accepts.

Every value `start_time` can take must lie inside the `[start_time, stop_time]`
window of `dynamic.solver_parameters`, checked when the config is loaded. For a
normal, a side with `min` or `max`, or the param's floor, ends there, and a
side without one ends 6 std beyond `mean`, or beyond the other side's bound
when that is further out. A draw never passes either end.
Each event must also fall by `stop_time`: the largest `start_time` plus the largest `delay` may
not exceed it, and for a `NodeFault` neither may that sum plus the largest
`fault_time`. The other types are instantaneous.

Each event variant draws from
`numpy.random.default_rng([settings.seed, scenario_index, perturbation_index,
event_index])`, so unless `topology_perturbation.type` is `random` a seed gives
the same events whatever `num_processes` and `large_chunk_size`. A random
topology perturbation draws per chunk, so the chunking changes the topology
variants and the targets placed on them.
The draws come in this order: the scenario by `weight`, only
when there are several; its targets; its `start_time`; then per event in list
order its `delay` and its `params`. A fixed value draws nothing, so adding
`delay: 0` changes no draw, and a `static_id` target draws nothing either. When
the drawn scenario cannot be placed the sample fails; no other scenario is
tried, which would bias the weights. The event variants of one topology variant
share its OPF and power flow: their Parquet snapshot rows are identical except `event_index`.

### `dynamic.logging` and `dynamic.validate` (optional)

```yaml
dynamic:
  validate: false          # run the static validation suite on the PF snapshot after the run
  logging:
    verbosity: info        # silent | error | warning | info | debug
    save_reports: true     # persist each simulation's Dynawo report
```

The values above are the defaults, so the whole block can be omitted.

`validate` is off by default: it re-reads every Parquet table, which is wasteful
on a large run. It covers only the **static snapshot**, the initial operating
point. The trajectories in the Zarr store are not validated.

`verbosity` sets the threshold of the `gridfm_datakit.dynamic` progress logger
only; it does not quieten the solvers, which `settings.enable_solver_logs`
governs. `silent` is the quietest setting but still lets errors through, since it
maps onto the same level as `error`.

### Settings that behave differently

The `settings:` block is the static one, with these caveats:

- `settings.large_chunk_size` bounds peak memory: a chunk is written and
  released before the next runs. Dynamic curves are far larger than static
  snapshots, so this matters more here than in the static pipeline.
- `settings.include_dc_res` is not honoured; the dynamic pipeline never computes
  DC results.
- `settings.pf_solver`, `settings.pf_fast` and `settings.dcpf_fast` are inert:
  OPF is always PowerModels and the initial-state AC power flow is always
  OpenLoadFlow.
- `settings.enable_solver_logs` routes OPF/PF **and** Dynawo's native output
  (OpenModelica banners, solver iterations) to `raw/solver_log/`. Turning it on
  also raises the Julia solver verbosity to DEBUG, which un-silences
  PowerModels; PowerModels logs through its own Julia-level logger, which the
  file router does not reliably capture, so `[ PowerModels | Info/Warn ]` lines
  can spill onto the console. The Dynawo per-simulation reports are saved
  regardless of this setting.
- `dynamic.output_dir` **no longer exists**. Outputs are rooted at
  `settings.data_dir` like the static pipeline; a config still carrying the key
  is rejected rather than silently writing somewhere unexpected.

### Full example

```yaml
network:
  name: IEEE14
  reader: powsybl
  source: file
  file: grids/IEEE14.iidm

load:
  generator: agg_load_profile
  agg_profile: default
  scenarios: 6
  sigma: 0.2
  change_reactive_power: true
  global_range: 0.4
  max_scaling_factor: 4.0
  step_size: 0.05
  start_scaling_factor: 0.8

topology_perturbation:
  type: none               # random | n_minus_k | none

dynamic:
  dynamic_solver: dynawo
  input_files:
    static_element_dynamic_models_file: inputs/static_element_dynamic_models.csv
    automation_systems_file: inputs/automation_systems.csv
    events_file: inputs/events.csv
    variables_file: inputs/variables.csv
  solver_parameters:
    start_time: 0.0
    stop_time: 500.0
    parameters_file: grids/IEEE14.par
    network_parameters_file: grids/IEEE14.par
    network_parameters_id: Network
    solver_type: SIM
    solver_parameters_file: grids/IEEE14.par
    solver_parameters_id: SimplifiedSolver
  validate: false
  logging:
    verbosity: info
    save_reports: true

settings:
  num_processes: 2
  data_dir: out
  large_chunk_size: 3
  overwrite: true
  mode: pf
  include_dc_res: false
  enable_solver_logs: false
  pf_fast: false
  dcpf_fast: false
  max_iter: 200
  pf_solver: powsybl
  seed: 42
```

## Running

A config carrying a `dynamic:` block selects the dynamic pipeline. There is no
separate subcommand:

```bash
gridfm_datakit generate path/to/dynamic_config.yaml
```

Note that the CLI does not resolve relative paths in the config against the
config's own folder, so paths must be absolute or relative to the working
directory.

From Python:

```python
from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data

file_paths = generate_dynamic_data("dynamic_config.yaml")  # or a dict / NestedNamespace
```

A ready-to-run end-to-end example lives in `scripts/dynamic_example/`
(IEEE14, generator `_GEN____2_SM` disconnected at `t = 50 s`). Its `run.py`
resolves the folder-relative paths and enables logging:

```bash
python scripts/dynamic_example/run.py
```

Progress is reported per chunk through the `gridfm_datakit.dynamic` logger:

```
HH:MM:SS INFO    gridfm_datakit.dynamic | Dynamic generation: 6 scenarios in 2 chunk(s), 2 worker(s).
HH:MM:SS INFO    gridfm_datakit.dynamic | Chunk 1/2 done (3 scenarios), 3 samples so far.
HH:MM:SS INFO    gridfm_datakit.dynamic | Chunk 2/2 done (3 scenarios), 6 samples so far.
HH:MM:SS INFO    gridfm_datakit.dynamic | Saved 6 samples to .../out/IEEE14/raw/dynamic (6 with dynamic results, 6 reports).
```

(That layout comes from `run.py`'s `logging.basicConfig`. With no application
logging configured, the library attaches its own handler and prefixes each line
with `[dynamic]` instead.)

## Perturbations

All the static perturbation blocks are accepted, but they do not contribute
equally to *dynamic* diversity:

| Block | Acts | Effect on the trajectory |
| --- | --- | --- |
| `load` scenarios | before OPF | Different loading, hence a different initial operating point |
| `generation_perturbation` | before OPF | **Does not work here**, see below |
| `admittance_perturbation` | before OPF | Perturbs branch `r`/`x`. These are pushed to pypowsybl together with the OPF set-points, so they change the network Dynawo simulates, not just the dispatch |
| `topology_perturbation` | before OPF | **The only one that changes which elements are in service**, hence the only one that changes the set of dynamic models Dynawo instantiates. One Dynawo run per perturbed topology |

### Why `generation_perturbation` does not work

It perturbs generator **cost** functions, and under `reader: powsybl`, which
dynamic runs require, there are none to perturb. pypowsybl does not carry cost
data in any format it reads, so every generator is loaded with the same
placeholder cost `(c2=0, c1=1, c0=0)`. Both strategies degenerate:

| `type` | Behaviour under `reader: powsybl` |
| --- | --- |
| `cost_permutation` | **Strict no-op.** It permutes cost rows that are all identical, so the dispatch is bit-for-bit unchanged. |
| `cost_perturbation` | Scales each coefficient by a random factor. `c2` and `c0` stay 0, `c1` becomes a random per-generator value, so the dispatch *does* change, but the spread is **synthetic**, drawn around a placeholder $1/MWh and unrelated to the network's real economics. |

!!! danger "A MATPOWER `.m` file's costs are silently discarded"
    This is the surprising case: `.m` files *do* carry a `gencost` block, but
    `reader: powsybl` drops it. The file is converted through pypowsybl, which
    has no cost concept, and the costs come back as the placeholder above.

    This is deliberate, not an oversight. pypowsybl gives no guarantee that the
    generator row order it produces matches the `.m` file's, so injecting the
    costs could attach them to the wrong generators. Neutral defaults were judged
    safer than silently wrong economics. The consequence is that a user who wrote
    real costs into their case file still gets the degenerate behaviour above.

Neither gives the cost diversity the block exists to provide. Use
`topology_perturbation` and the load scenarios for dynamic diversity instead.

`admittance_perturbation` is wired for parity with the static pipeline. Its
perturbed line and transformer impedances do reach the simulated network,
since `update_powsybl` writes `r` and `x` alongside the OPF set-points, so it
moves the trajectory and not only the dispatch. Its effect on the dynamic outputs is
nonetheless unvalidated, and unlike `topology_perturbation` it yields one sample
per scenario rather than expanding it.

Only `topology_perturbation` expands one load scenario into several samples,
which is why `perturbation_index` exists at all. Each perturbation runs in its
own pypowsybl network variant, and a failing perturbation is logged and dropped
without taking the rest of the scenario with it.

## Outputs

Everything a run produces lives under one root, `settings.data_dir`, reusing the
static pipeline's `{data_dir}/{network.name}/raw/`:

```
{settings.data_dir}/{network.name}/raw/
├── args.log, error.log
├── scenarios_{generator}.{parquet,html,log}
├── solver_log/                        only when settings.enable_solver_logs
└── dynamic/
    ├── bus_data.parquet
    ├── branch_data.parquet
    ├── gen_data.parquet
    ├── y_bus_data.parquet
    ├── runtime_data.parquet
    ├── final_state_values.parquet     only when FinalStateValue rows are monitored
    ├── events.parquet
    ├── dynamic_results.zarr/
    ├── reports/
    └── metadata.json
```

The dynamic artifacts sit in a `dynamic/` subfolder rather than directly in
`raw/` because the static pipeline writes `bus_data.parquet` as a *partitioned
directory* while the dynamic pipeline writes it as a flat file: same name,
different kind, so they must not share a directory. `raw/dynamic/` is owned by
the pipeline and recreated on every run, so it never mixes fresh artifacts with
a previous run's leftovers.

### Static snapshot (Parquet)

The initial operating point Dynawo started from, in the same column schemas as
the static pipeline ([Outputs](outputs.md)) with two differences:

- The files are **flat Parquet files**, not partitioned directories.
- The static pipeline's `scenario` / `load_scenario_idx` columns are replaced by
  the triple **`scenario_index`, `perturbation_index`, `event_index`**, inserted
  as the first three columns. `load_scenario_idx` cannot tell two topology
  perturbations of one load scenario apart; this triple can.

Datasets written by release 1.1.0 and earlier carry only `scenario_index` and
`perturbation_index`; `validate_dynamic_data` reads them as `event_index` 0.

### Trajectories (`dynamic_results.zarr`)

| Array | Shape | Contents |
| --- | --- | --- |
| `curves` | `(n_samples_with_curves, n_variables, n_timesteps)` | The monitored `Curve` variables, NaN-padded along the time axis |
| `time` | `(n_samples_with_curves, n_timesteps)` | Simulation time in **seconds**, per sample, NaN-padded to match |
| `scenario_index` | `(n_samples_with_curves,)` | Join key: the load scenario each slice came from |
| `perturbation_index` | `(n_samples_with_curves,)` | Join key: the topology perturbation each slice came from |
| `event_index` | `(n_samples_with_curves,)` | Join key: the event variant each slice came from |

Axis 0 is the number of samples that produced curves, reported in
`metadata.json` as `n_samples_with_curves`. Take the length from there rather
than from `n_samples`, which counts the samples written to the Parquet snapshot.

Variable names are the flattened `<model_id>_<variable>` names pypowsybl
returns (e.g. `_GEN____1_SM_generator_efdPu_value`), listed in order in
`metadata.json` under `variable_names`. Dynawo does **not** return variables in
registration order, so read the names rather than assuming the order of the
input table. The first sample with curves establishes the stored variable
order. Later samples with the same unique names are reordered to match it;
missing, unexpected, or duplicate names raise `ValueError` before any part of
the chunk is appended. If a topology perturbation removes an element whose
curve is monitored, the returned variable-name set changes; `write_chunk`
raises and the generation run stops rather than dropping only that sample.

!!! note "Never assume a shared time axis"
    A variable-step solver (`solver_type: IDA`) gives each run its own time grid
    and its own number of timesteps. The store is sized to the longest run seen;
    shorter samples keep the NaN fill, and their valid length is recorded per
    sample in `metadata.json` under `timesteps_per_scenario`. Read the `time`
    array instead of reconstructing a uniform grid from `start_time`/`stop_time`.

### `final_state_values.parquet`

One row per sample, keyed by `(scenario_index, perturbation_index, event_index)`,
with one
column per monitored `FinalStateValue` variable. Written only when the run
declares such rows. The column set is fixed by the first chunk that carries
values; a later sample reporting a different set is reindexed onto it (unknown
names dropped, missing ones become `NaN`).

### `events.parquet`

One row per event per sample, keyed by `(scenario_index, perturbation_index,
event_index)`, with the columns `scenario`, `event_name`, `static_id`,
`start_time` and `params`. It records the events each sample simulated, read
from `events_file` or drawn. `scenario` is the name of the drawn scenario, `""`
for `events_file` rows. With `event_perturbation.type: none` no file is written.

### `reports/`

One JSON file per sample, `scenario_{i}_perturbation_{j}_event_{k}.json`, holding
pypowsybl's `ReportNode`, covering the model build-up and problem resolution. This is
the documented way to diagnose a failed or degenerate run. Controlled by
`dynamic.logging.save_reports`. Report verbosity can be raised through the
Dynawo simulation parameter `log.levelFilter`.

### `metadata.json`

| Key | Meaning |
| --- | --- |
| `generated_at`, `seed`, `config_hash` | Provenance |
| `n_samples` | Number of **samples**, not load scenarios: larger than `load.scenarios` as soon as a scenario expands into more than one topology, smaller when samples failed |
| `n_samples_with_curves` | Length of axis 0 of `curves` |
| `variable_names`, `n_variables` | Axis 1 of `curves`, in order |
| `n_timesteps`, `timesteps_per_scenario`, `time_units` | Axis 2 of `curves`; the per-sample valid (unpadded) length |
| `static_scenario_index`, `static_perturbation_index`, `static_event_index` | Join keys present in the Parquet snapshot |
| `dynamic_scenario_index`, `dynamic_perturbation_index`, `dynamic_event_index` | Join keys present in the Zarr store |
| `final_state_value_names` | Columns of `final_state_values.parquet`, in order |
| `reports` | Report file names |

### Joining features to labels

A sample reaches the outputs only when every step succeeded: a failed OPF, a
failed power flow or a failed Dynawo run drops the whole sample, its static rows
included. So a curves slice index is **not** a scenario number: failed samples
leave gaps, and a run with `topology_perturbation` has several slices per load
scenario. **Always join on the `(scenario_index, perturbation_index, event_index)`
key triple, never on row or slice position.**

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

root = Path("out/IEEE14/raw/dynamic")
meta = json.loads((root / "metadata.json").read_text())
store = zarr.open(str(root / "dynamic_results.zarr"), mode="r")
bus = pd.read_parquet(root / "bus_data.parquet")

keys = zip(*(np.asarray(store[k]) for k in ("scenario_index", "perturbation_index", "event_index")))
slice_of = {tuple(int(v) for v in key): i for i, key in enumerate(keys)}

i = slice_of[(0, 0, 0)]
n = meta["timesteps_per_scenario"][i]          # drop the NaN padding
t = store["time"][i, :n]                       # seconds
u = store["curves"][i, meta["variable_names"].index("_BUS____2_TN_U_value"), :n]
initial_state = bus[
    (bus.scenario_index == 0) & (bus.perturbation_index == 0) & (bus.event_index == 0)
]
```

## Failure handling

Failures are contained at the smallest scope that makes sense, so one bad sample
never aborts a run. Per-event-variant, per-perturbation and per-scenario failures
are appended to
`raw/error.log` with their traceback. A worker that dies before reaching its
scenario loop never gets to write there, so the parent reports it through the
progress logger instead:

| Failure | Behaviour | Reported in |
| --- | --- | --- |
| One event variant fails (placement or simulation) | The topology variant's other event variants continue | `raw/error.log`, as `scenario i perturbation j event k failed` |
| One topology perturbation fails | The scenario's other perturbations continue | `raw/error.log` |
| One scenario fails | The rest of the chunk continues | `raw/error.log` |
| A whole worker dies | The other workers and chunks continue | progress logger, as `Error in dynamic chunk: …` |
| **Every** sample fails | `close()` raises `RuntimeError` | the raised error |

When every sample fails, `raw/dynamic/` is never created, since the writer only
makes it on the first chunk that carries a sample, so no half-written output is
left behind. Whether the *previous* run's data survives is decided earlier and
elsewhere: `settings.overwrite: true` deletes the whole `{data_dir}/{name}/raw/`
tree at startup, before any simulation runs.

Two silent failure modes are turned into hard errors on purpose, because Dynawo
reports neither:

- **A failed simulation is not raised by Dynawo.** `sim.run()` returns a result
  whose status is `FAILURE` with empty or truncated curves. The pipeline checks
  the status and drops the sample instead of storing a diverged run as a valid
  trajectory.
- **A model Dynawo cannot instantiate is skipped, and the run still reports
  `SUCCESS`.** The sample would then be a trajectory of a *different system*
  than the input tables describe. The pipeline parses the report for failed
  instantiations and raises, naming the offending models.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| `Dynawo backend unavailable: …` | No `~/.itools/config.yml`, no `dynawo.homeDir` entry, or `homeDir` does not contain `dynawo.sh` / `bin/dynawo` (`dynawo.cmd` on Windows) |
| `Dynamic simulations require network.reader='powsybl'` | Set `reader: powsybl` in the `network` block |
| `Dynawo failed to instantiate N dynamic model(s)` | A `static_id`, `model_name` or `category_name` does not match the network's element IDs or a Dynawo model. Check the IDs against the network file, not against `get_*()` output |
| `variables: no row of type 'Curve'` | The time-series store needs at least one `Curve` row |
| `automation_systems: unsupported category_name …` | Typo in a key column; the message lists the accepted values |
| `dynamic.solver_parameters: missing required key(s)` | `start_time` / `stop_time` are mandatory |
| `Error in dynamic chunk: …` on every chunk | A per-worker setup step failed identically everywhere, most often a bad key in `dynamic.solver_parameters` or `dynamic.loadflow_parameters`. The message carries the underlying error |
| `Dynamic generation produced no samples: every scenario failed` | Read `raw/error.log` for the per-scenario cause, and the `Error in dynamic chunk` lines above it for worker-level ones. No reports are written in this case, since `raw/dynamic/` is never created |
| Trajectory looks like the base case | Check that `events.csv` `start_time` falls inside `[start_time, stop_time]` |

## Current limitations

- Dynawo is the only backend. `dynamic_solver` is the extension point, but any
  other value raises `NotImplementedError`.
- The dynamic model set and automation systems are the same for every sample,
  and so are the events unless `event_perturbation` draws them. What varies is
  the operating point, the branch impedances (with `admittance_perturbation`)
  and the topology (with `topology_perturbation`).
- Dynawo ignores a power or voltage variation when the target's dynamic model
  does not take it: the run succeeds and the event is recorded, but the curves
  do not change. On IEEE14 this is `ActivePowerVariation` and
  `ReactivePowerVariation` on the loads modelled as
  `LoadOneTransformerTapChanger` (`_LOAD___6_EC` and `_LOAD___9_EC` in the
  example) or `LoadTwoTransformersTapChangers` (`_LOAD___2_EC` in the test
  configuration), `ReactivePowerVariation` on every synchronous generator, and
  `ReferenceVoltageVariation` on `_GEN____3_SM`.
- The event graph joins buses by lines and transformers only. Bus-breaker buses
  linked only by a closed switch, a tie line or an HVDC line are placed as if
  apart, and a bus reached only through such links is unreachable from the
  anchor. Tie lines and HVDC lines are never `line` targets.
- A bus of the network file with no in-service branch is still a random anchor,
  around which only distance 0 can be placed, and a `bus` target. The topology
  perturbation never leaves such a bus, since it keeps connected variants only.
- `events_file` times are not checked against the simulation window.
- `generation_perturbation` does not work: the powsybl reader supplies no real
  generator costs for it to perturb.
- The `TapChangerBlocking` automation system cannot be configured from the CSV
  inputs; its parameters are DataFrames and the `params` column holds scalars.
- `dynamic.validate` covers the static snapshot only; the trajectories are not
  validated.
- The CLI `validate`, `stats` and `plots` commands read the static pipeline's
  partitioned layout and do not accept `raw/dynamic/`.
- All samples in a run must monitor the same variables, since they share one Zarr
  store, and a mismatch is rejected rather than written as a corrupt array.
