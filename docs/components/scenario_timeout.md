# Scenario timeouts

To skip load scenarios that hang or exceed a wall-clock limit, use the Pool
backend and set a positive timeout in seconds:

```yaml
settings:
  process_backend: multiprocessing_pool
  scenario_timeout_sec: 600
  enable_checkpointing: true
```

The timer covers one complete load scenario: its dispatch OPF and all topology
variants. It begins after worker/solver initialization, immediately before
scenario processing. Initialization, queue wait, and result transfer are outside
this limit. A scenario that exceeds it is skipped in full, so none of its
partially accumulated variants are saved.

`ScenarioTimeoutPool` supervises workers from the parent process. The main
process kills the worker that ran too long, stops the other workers, and runs
the unfinished chunk again. Scenarios that already timed out are left out. Other
scenarios from that attempt are recomputed. Already checkpointed chunks remain
untouched. Retries finish even if every scenario in a chunk times out. Ordinary
exceptions and unexpected worker exits stop generation rather than being silently
skipped.

Skipped global `load_scenario_idx` values and measured durations are written to
`timed_out_scenarios.jsonl` and reported in `error.log`.
With [checkpointing](checkpoint.md) enabled, the checkpoint also persists
`skipped_scenarios` immediately, before the chunk is
saved, so resuming does not rerun known offenders. Without checkpointing, the skip
set applies to the current invocation. Logs retain the history of aborted attempts.

If nothing times out, the data matches a run with no timeout. If a scenario is
skipped, it uses no random numbers, so the later scenarios in that chunk can differ.
Keep the timeout setting unchanged when resuming a checkpointed run.

The default `scenario_timeout_sec: null` disables timeouts. Timeouts apply only
to distributed static generation with
`process_backend: multiprocessing_pool`; sequential and dynamic pipelines reject them.

::: gridfm_datakit.process.scenario_timeout.ScenarioTimeout

::: gridfm_datakit.process.scenario_timeout.ScenarioTimeoutPool

::: gridfm_datakit.process.scenario_timeout._collect_timed_results
