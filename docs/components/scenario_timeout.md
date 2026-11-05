# Scenario timeouts

The Pool backend uses this supervisor when `settings.scenario_timeout_sec` is set.
See the [timeout manual](../manual/getting_started.md#scenario-timeouts) for timer
boundaries, retry behavior, skipped-index reporting, and seed behavior.

::: gridfm_datakit.process.scenario_timeout.ScenarioTimeout

::: gridfm_datakit.process.scenario_timeout.ScenarioTimeoutPool

::: gridfm_datakit.process.scenario_timeout._collect_timed_results
