# Network

The network parameters are the following:

```yaml
network:
  name: "case24_ieee_rts" # Name of the power grid network (without extension)
  source: "pglib" # Data source for the grid; options: pglib, pandapower, file
  network_dir: "scripts/grids" # if using source "file", this is the directory containing the network file (relative to the project root)

```

Networks can be loaded from three different sources, specified in `source:

## [PGLib repository](https://github.com/power-grid-lib/pglib-opf) (recommended)

e.g.
```yaml
network:
  source: "pglib"
  name: "case24_ieee_rts"   # Name of the power grid network **without the pglib prefix**
```

##  [Pandapower library](https://pandapower.readthedocs.io/en/v2.3.0/networks.html)

e.g.
```yaml
network:
  source: "pandapower"
  name: "case_ieee30"
```

## Local matpower files

e.g.
```yaml
network:
  source: "Texas2k_case1_2016summerpeak"
  name: "case24_ieee_rts"          # Name of the power grid network **without .m extension**
  network_dir: "scripts/grids"    # Directory containing the network files
```
