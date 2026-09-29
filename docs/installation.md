# Installation

1. ⭐ Star the [repository](https://github.com/gridfm/gridfm-datakit) on GitHub to support the project!

2. Make sure you have Python 3.10, 3.11, or 3.12 installed. ⚠️ Windows users: Python 3.12 is not supported. Use Python 3.10.11 or 3.11.9.

3. Install gridfm-datakit

```bash
python -m pip install --upgrade pip  # Upgrade pip
pip install gridfm-datakit
```

4. Install Julia with PowerModels and Ipopt
```bash
gridfm_datakit setup_pm
```

### Optional: lightsim2grid power flow solver

The `lightsim2grid` extra installs [lightsim2grid](https://github.com/Grid2op/lightsim2grid), an
alternative to PowerModels for solving the power flow (`settings.pf_solver: lightsim2grid`, see
[Power flow solver](manual/power_flow_solver.md)):

```bash
pip install 'gridfm-datakit[lightsim2grid]'
```

!!! warning "In the meantime: install lightsim2grid from source"
    The methods that let gridfm-datakit update the impedances of the lightsim2grid model in
    place (`update_powerlines_parameters` and `update_trafos_parameters`) are not in a
    lightsim2grid release yet. They are on the
    [`dev_gfm_datakit`](https://github.com/grid2op/lightsim2grid/tree/dev_gfm_datakit) branch.
    Until a release has them, install lightsim2grid from that branch, then gridfm-datakit:

    ```bash
    pip install 'lightsim2grid @ git+https://github.com/grid2op/lightsim2grid.git@dev_gfm_datakit'
    pip install 'gridfm-datakit[lightsim2grid]'
    ```

    This compiles C++ code, so it needs a C++ compiler and takes a few minutes. Without these
    methods `pf_solver: lightsim2grid` still gives the same results, but the model is rebuilt for
    every power flow, which is slower. Once a release contains them, the extra alone will be
    enough and this note will be removed.

### Optional: dynamic (time-domain) simulation

Dynamic simulation needs two extra things on top of the base install. See the
[Dynamic Simulation](manual/dynamic_simulation.md) page for the full setup.

1. The `dynamic` extra, which pulls in `pypowsybl` and `zarr`:

    ```bash
    pip install 'gridfm-datakit[dynamic]'
    ```

2. A local [Dynawo](https://dynawo.github.io/install/) installation. It is a
   native solver and is **not** bundled with pypowsybl. Extract a release (e.g.
   to `~/dynawo`) and declare it to powsybl in `~/.itools/config.yml`:

    ```yaml
    dynawo:
      homeDir: /path/to/dynawo
      debug: false
    ```

Verify both are usable:

```bash
python -c "from gridfm_datakit.dynamic.dynawo.api import check_dynawo_available; check_dynawo_available()"
```

### For Developers

To install the latest development version from GitHub, follow these steps instead of step 3.


```bash
git clone https://github.com/gridfm/gridfm-datakit.git
cd "gridfm-datakit"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip  # Upgrade pip to ensure compatibility with pyproject.toml
pip3 install -e '.[test,dev]'
```
