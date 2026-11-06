# Configuration

This module validates configuration for static PF and OPF generation.

### Checkpointing

`settings.enable_checkpointing` defaults to `false`. When enabled, distributed
static generation persists its seed and completed chunks in `checkpoint.json`.
Use `overwrite: false` to resume with matching settings. See the
[checkpoint and resume manual](../manual/getting_started.md#checkpoint-and-resume).

### Multiprocessing backend

Set `settings.is_multiprocessing: true` to select a spawned `multiprocessing.Pool`
with a Manager queue for live scenario progress. The default is `false`, which
keeps the existing `ProcessPoolExecutor` backend. Both use the same chunk
boundaries, seed policy, solver options, and output ordering. See the
[generation manual](../manual/getting_started.md#parallel-generation).

### CoinHSL

To use an installed CoinHSL library with Ipopt, add the following under `settings`:

```yaml
settings:
  coinhsl:
    enabled: true
    linear_solver: "ma57"
    hsllib: "/path/to/libcoinhsl.so"
```

CoinHSL is disabled by default. Supported solver names are `ma27`, `ma57`,
`ma77`, `ma86`, and `ma97`; the default is `ma57`. Supply the shared-library
file for your platform. Relative paths resolve from the working directory.
When enabled, the library file must exist before generation starts; Ipopt
checks library compatibility and availability of the selected solver at solve time.
The library must also be accessible to each worker process.

The settings apply to Ipopt-based AC/DC OPF and non-fast AC/DC PF solves,
including the OPF search used to determine load scaling and the initial OPF
for dynamic generation. Fast PF computations and the PowSyBl, LightSim2Grid,
and Dynawo engines keep their own solvers.

For direct Python initialization, pass the same mapping as
`init_julia(max_iter=200, coinhsl={"enabled": True, "linear_solver": "ma57", "hsllib": "/path/to/libcoinhsl.so"})`.

### `validate_static_config`

::: gridfm_datakit.config.validate_static_config

### `CoinHSLConfig`

::: gridfm_datakit.config.CoinHSLConfig
