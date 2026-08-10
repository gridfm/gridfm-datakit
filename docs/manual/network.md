# Network

The network parameters are the following:

```yaml
network:
  name: "case24_ieee_rts" # Name of the power grid network (without extension)
  source: "pglib" # Data source for the grid; options: pglib, file
  network_dir: "scripts/grids" # if using source "file", this is the directory containing the network file (relative to the project root)

```

Networks can be loaded from two different sources, specified in `source:

## [PGLib repository](https://github.com/power-grid-lib/pglib-opf) (recommended)

e.g.
```yaml
network:
  source: "pglib"
  name: "case24_ieee_rts"   # Name of the power grid network **without the pglib prefix**
```

## Local MATPOWER files

e.g.
```yaml
network:
  source: "file"
  name: "Texas2k_case1_2016summerpeak"  # Name of the power grid network **without .m extension**
  network_dir: "scripts/grids"          # Directory containing the network files
```

## Choosing the reader

`reader` controls **how** the file is parsed, independently of `source` (which
controls *where* it comes from):

- `native` (default): the built-in MATPOWER reader.
- `powsybl`: parsed by pypowsybl, which additionally supports XIIDM, CGMES,
  PSS/E (`.raw`), UCTE (`.uct`) and MATPOWER binary (`.mat`). Requires
  `pip install 'gridfm-datakit[powsybl]'`.

With `reader: powsybl` and `source: file`, the optional `file` key points
directly at the network file (extension included) and takes precedence over
`network_dir` + `name`:

```yaml
network:
  name: "IEEE14"
  reader: "powsybl"
  source: "file"
  file: "grids/IEEE14.iidm"
```

`reader: powsybl` is **required** for
[dynamic simulation](dynamic_simulation.md).

!!! note "Generator costs are defaulted under `reader: powsybl`"
    pypowsybl does not carry generator cost functions in any format it reads, so
    networks loaded this way get neutral defaults (`c2=0, c1=1, c0=0`).
