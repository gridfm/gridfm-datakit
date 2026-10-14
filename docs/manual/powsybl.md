# PowSyBl

[PowSyBl](https://www.powsybl.org/) can read the grid and solve the power flow. The optimal power flow always runs in PowerModels, so Julia is still needed.

```bash
pip install 'gridfm-datakit[powsybl]'
gridfm_datakit setup_pm
```

## Reading the grid

`network.reader: powsybl` reads the grid with pypowsybl. This is how to load XIIDM, CGMES, PSS/E (`.raw`), UCTE (`.uct`) and MATPOWER files.

- For a local file, set `source: file` and put the path to the file, extension included, in `file`.
- For a PGLib case, set `source: pglib` and put the case name without the `pglib_opf_` prefix in `name`.

## Solving the power flow

`settings.pf_solver: powsybl` solves the power flow with [Open Load Flow](https://powsybl.readthedocs.io/projects/powsybl-open-loadflow/) instead of PowerModels. The generator set-points still come from the PowerModels optimal power flow. It requires `mode: pf` and `reader: powsybl`, and has no effect in `mode: opf`.

The default, `pf_solver: powermodel`, works with both readers.

## Generator costs

PowSyBl does not read generator costs, including from PGLib. Every generator is given the same cost, `c2=0`, `c1=1`, `c0=0`, so the optimal power flow has no preference for one generator over another.

`cost_permutation` has no effect, since all costs are identical. `cost_perturbation` gives each generator its own random cost, so the dispatch varies, but not according to the real costs of the grid. Use `none`.

## Examples

Only the keys that differ from `scripts/config/default.yaml` are shown.

A PGLib case

```yaml
network:
  name: "case24_ieee_rts"
  source: "pglib"
  reader: "powsybl"

generation_perturbation:
  type: "none"

settings:
  mode: "pf"
  pf_solver: "powsybl"
```

A local PSS/E file

```yaml
network:
  name: "my_case"
  source: "file"
  reader: "powsybl"
  file: "grids/my_case.raw"

generation_perturbation:
  type: "none"

settings:
  mode: "pf"
  pf_solver: "powsybl"
```
