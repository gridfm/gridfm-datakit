# Power flow solver

In `mode: "pf"` the power flow of every perturbed network is solved by the engine chosen with
`settings.pf_solver`. The OPF that gives the generator set points is **always** solved by
PowerModels.jl, whatever the value of this parameter.

```yaml
settings:
  pf_solver: "lightsim2grid" # options: powermodel (default), powsybl, lightsim2grid
```

| `pf_solver` | Engine | Extra to install |
|---|---|---|
| `powermodel` (default) | PowerModels.jl, through Julia. `pf_fast` and `dcpf_fast` choose between the direct and the optimizer-based solvers | none |
| `powsybl` | pypowsybl (Open Load Flow). Needs the network to be loaded with `network.reader: powsybl` | `pip install 'gridfm-datakit[powsybl]'` |
| `lightsim2grid` | [lightsim2grid](https://github.com/Grid2op/lightsim2grid) (C++, no Julia for the power flow) | `pip install 'gridfm-datakit[lightsim2grid]'` |

## lightsim2grid

The network is read with the native reader, and only the power flow solver changes:

```yaml
network:
  source: "pglib"
  name: "case118_ieee"

settings:
  mode: "pf"
  pf_solver: "lightsim2grid"
  include_dc_res: true # the DC power flow is solved by lightsim2grid too
```

### How the network is converted

The lightsim2grid model is built straight from the bus, generator and branch arrays of the
[`Network`](../components/network.md), so it is the network that the perturbations have
modified (loads, admittances, generator set points, outages). Buses and generators keep their order,
and each branch becomes a powerline or a transformer (a branch with a non zero tap or phase
shift is a transformer).

The model is not rebuilt for every perturbation: what changed since the previous power flow
(branch impedances and statuses, generator statuses and set points, loads) is pushed into the same
model, which lets lightsim2grid keep its solver state. It is rebuilt only if something that cannot
be pushed changed (the network topology, a tap ratio or a phase shift, the shunts, the bus types,
which buses have a load, or the slack generators). The model is kept by each worker process, so it
is also reused from one scenario to the next.

!!! note "Versions"
    `pf_solver: lightsim2grid` needs a lightsim2grid providing
    `lightsim2grid.network.init_from_matpower`. Pushing changes into the model needs the
    `update_powerlines_parameters` and `update_trafos_parameters` methods; without them the
    model is rebuilt for every power flow, which gives the same results more slowly. Until a
    lightsim2grid release has them, they are available by
    [installing lightsim2grid from source](../installation.md#optional-lightsim2grid-power-flow-solver).

### Differences with PowerModels

- **AC**: the results agree with those of the fast PowerModels power flow (`pf_fast: true`). On
  case14 the difference is below `1e-6` on all the bus, generator and branch quantities.
- **DC**: the two solvers do not use the same DC model. lightsim2grid uses the MATPOWER one
  (susceptance `1 / (x * tap)`) whereas PowerModels uses `x / (r^2 + x^2)`, so the DC angles and
  flows differ, more on networks with large resistances.
- **Slack**: the active power imbalance is shared equally by the in-service generators
  connected to the reference bus.
- **No optimizer**: there is no equivalent of `pf_fast: false`, and `pf_fast` / `dcpf_fast`
  are ignored.
