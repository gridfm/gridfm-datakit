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
