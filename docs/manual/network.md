# Network

The network parameters are the following:

```yaml
network:
  source: "pglib"                  # Data source; options: pglib, pandapower, file
  name: "case24_ieee_rts"          # Name of the power grid network
  network_dir: "grids"             # Directory containing the network files
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
