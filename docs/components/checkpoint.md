# Checkpoint and resume

Enable checkpoints for distributed static generation (either process backend):

```yaml
settings:
  enable_checkpointing: true
  overwrite: true
  seed: null
```

`GenerationCheckpoint` manages the saved state. The run writes `checkpoint.json`
before generating scenarios. It saves the actual random seed even when `seed`
is null, then advances the checkpoint only after all
five output datasets for a large chunk are saved. To resume an interrupted run,
use the same configuration and output directory with `overwrite: false`.
The saved seed is reused; completed large chunks are skipped. An already completed
run returns its artifact paths without regenerating scenarios or starting workers.

The checkpoint records the committed parquet files and output sample count. On
resume, parquet files written after the last committed checkpoint are removed and
that count is restored, allowing an interrupted chunk save to be retried without
duplicate samples. Missing or truncated committed files cause an error.

All validated configuration settings must match except `overwrite`, `data_dir`,
`enable_solver_logs`, and `process_backend`, with the seed handled as described
below. In particular, keep the same scenario count, worker count, chunk size,
network, perturbations, solver settings, and `scenario_timeout_sec`.
An explicit seed must match the saved seed; `seed: null` uses the saved one.
You may change solver logging or move the complete output
directory. You may switch process backends when [scenario timeouts](scenario_timeout.md)
are disabled. Keep input-file contents and the software environment unchanged
for reproducible results, and use one generation process
per output directory.

Existing data without a checkpoint cannot be adopted into a resumable run.
Disabling checkpointing on existing checkpointed data is rejected to prevent an
ordinary append from invalidating its state. Use `overwrite: true` for a fresh
run. Checkpointing is disabled by default and applies to the CLI's static pipeline
and `generate_power_flow_data_distributed`, not the sequential or dynamic APIs.

::: gridfm_datakit.checkpoint.GenerationCheckpoint
