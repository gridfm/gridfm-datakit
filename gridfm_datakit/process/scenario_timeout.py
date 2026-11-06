"""Monitor load scenarios and restart a Pool after a scenario deadline expires."""

from multiprocessing.context import BaseContext
from multiprocessing.pool import Pool
from queue import Empty
import time
from typing import Any, Callable, Optional

import numpy as np
from tqdm import tqdm


class ScenarioTimeout(TimeoutError):
    """Identify a load scenario that exceeded its wall-clock limit.

    Attributes:
        scenario_index: Global load-scenario index.
        worker_pid: Process running the scenario.
        elapsed: Measured scenario duration in seconds.
    """

    def __init__(self, scenario_index: int, worker_pid: int, elapsed: float) -> None:
        """Store the offender and its measured duration.

        Args:
            scenario_index: Global load-scenario index.
            worker_pid: Process running the scenario.
            elapsed: Measured scenario duration in seconds.
        """
        self.scenario_index = scenario_index
        self.worker_pid = worker_pid
        self.elapsed = elapsed
        super().__init__(
            f"Load scenario {scenario_index} timed out after {elapsed:.3f}s (pid={worker_pid})",
        )


def _collect_timed_results(
    pool: Pool,
    tasks: list[tuple[int, np.ndarray]],
    progress_queue: Any,
    progress_bar: tqdm,
    timeout: float,
    worker: Callable,
) -> dict[int, tuple]:
    """Collect chunk results while enforcing deadlines on active load scenarios.

    Args:
        pool: Spawned pool using timed worker events.
        tasks: Global offsets and contiguous scenario slices.
        progress_queue: Queue receiving start/done/error events.
        progress_bar: Parent-process progress display.
        timeout: Maximum wall-clock seconds per active load scenario.
        worker: Picklable scenario worker callable.

    Returns:
        Results keyed by their submitted global offsets.

    Raises:
        ScenarioTimeout: If a running or completed scenario exceeds its limit.
        RuntimeError: If a worker fails, exits unexpectedly, or emits invalid events.
    """
    pending = {
        start: pool.apply_async(worker, ((start, scenarios),))
        for start, scenarios in tasks
    }
    workers = tuple(pool._pool)
    active = {}
    results = {}
    completed = 0

    def event_received(event: tuple) -> None:
        """Update active deadlines from one ordered worker event.

        Args:
            event: Event kind, worker PID, global index, and monotonic timestamp.
        """
        nonlocal completed
        kind, pid, index, timestamp = event
        if kind == "start":
            active[pid] = (index, timestamp)
        elif kind == "done":
            if pid not in active or active[pid][0] != index:
                raise RuntimeError(f"Unexpected scenario completion event: {event!r}")
            _, started = active.pop(pid)
            elapsed = timestamp - started
            if elapsed > timeout:
                raise ScenarioTimeout(index, pid, elapsed)
            completed += 1
            progress_bar.update(1)
        elif kind == "error":
            raise RuntimeError(
                f"Scenario worker pid={pid} failed at load scenario {index}; see error.log",
            )
        else:
            raise RuntimeError(f"Unknown scenario progress event: {event!r}")

    def read_events(block: bool) -> None:
        """Drain queued completions before inspecting active deadlines.

        Args:
            block: Whether to wait briefly for the first available event.
        """
        try:
            event_received(progress_queue.get(timeout=0.05 if block else 0))
        except Empty:
            return
        while True:
            try:
                event_received(progress_queue.get_nowait())
            except Empty:
                return

    try:
        while pending:
            read_events(True)
            for start in list(pending):
                if not pending[start].ready():
                    continue
                try:
                    result = pending.pop(start).get()
                except Exception as error:
                    raise RuntimeError(
                        f"Scenario worker failed for chunk starting at {start}",
                    ) from error
                if result[2] is not None:
                    raise RuntimeError(
                        f"Scenario chunk starting at {start} failed: {result[2]}\n{result[3]}",
                    ) from result[2]
                results[start] = result
            for process in workers:
                if process.exitcode is not None:
                    raise RuntimeError(
                        f"Scenario worker pid={process.pid} exited unexpectedly (exit code {process.exitcode})",
                    )
            # A done event may arrive while large results are being collected.
            read_events(False)
            now = time.monotonic()
            for pid, (index, started) in active.items():
                if now - started > timeout:
                    raise ScenarioTimeout(index, pid, now - started)
        read_events(False)
        return results
    except BaseException:
        if completed:
            progress_bar.update(-completed)
        raise


class ScenarioTimeoutPool:
    """Reuse a Pool until a timeout, then skip the offender and retry its chunk.

    Attributes:
        skipped_scenarios: Global indices excluded from this and future attempts.
    """

    def __init__(
        self,
        context: BaseContext,
        processes: int,
        initializer: Callable,
        initargs: tuple,
        worker: Callable,
        timeout: float,
        skipped_scenarios: set[int],
        on_timeout: Callable[[ScenarioTimeout], None],
    ) -> None:
        """Configure pool ownership and durable timeout reporting.

        Args:
            context: Spawn multiprocessing context.
            processes: Maximum worker count.
            initializer: Existing scenario-worker initializer.
            initargs: Shared worker state before queue/timeout/skip arguments.
            worker: Existing scenario-worker callable.
            timeout: Per-load-scenario wall-clock limit in seconds.
            skipped_scenarios: Previously persisted timed-out global indices.
            on_timeout: Parent callback persisting each newly skipped index.
        """
        self.context = context
        self.processes = processes
        self.initializer = initializer
        self.initargs = initargs
        self.worker = worker
        self.timeout = timeout
        self.skipped_scenarios = set(skipped_scenarios)
        self.on_timeout = on_timeout
        self.manager = None
        self.pool: Optional[Pool] = None
        self.progress_queue = None

    def __enter__(self) -> "ScenarioTimeoutPool":
        """Start the Manager; workers start only when a non-skipped task needs them."""
        self.manager = self.context.Manager()
        return self

    def _start_pool(self) -> None:
        """Start fresh workers and a fresh queue, excluding all known offenders."""
        self.progress_queue = self.manager.Queue()
        self.pool = self.context.Pool(
            processes=self.processes,
            initializer=self.initializer,
            initargs=self.initargs
            + (self.progress_queue, self.timeout, set(self.skipped_scenarios)),
        )

    def _stop_pool(self, offending_pid: Optional[int] = None) -> None:
        """Stop workers before reusing any results or progress state.

        Args:
            offending_pid: Worker to kill before terminating the remaining pool.
        """
        if self.pool is None:
            return
        if offending_pid is not None:
            for process in self.pool._pool:
                if process.pid == offending_pid and process.is_alive():
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    break
        self.pool.terminate()
        self.pool.join()
        self.pool = None

    def collect(
        self, tasks: list[tuple[int, np.ndarray]], progress_bar: tqdm,
    ) -> dict[int, tuple]:
        """Retry an unsaved large chunk until every non-skipped scenario finishes.

        Args:
            tasks: Original chunk offsets and slices, preserved across retries.
            progress_bar: Parent progress display, including skipped scenarios.

        Returns:
            Ordered-by-offset-compatible task results for the successful attempt.
        """
        indices = {
            index
            for start, scenarios in tasks
            for index in range(start, start + scenarios.shape[1])
        }
        progress_bar.update(len(indices & self.skipped_scenarios))
        while True:
            if indices <= self.skipped_scenarios:
                return {
                    start: (start, scenarios.shape[1], None, None, [])
                    for start, scenarios in tasks
                }
            if self.pool is None:
                self._start_pool()
            try:
                return _collect_timed_results(
                    self.pool,
                    tasks,
                    self.progress_queue,
                    progress_bar,
                    self.timeout,
                    self.worker,
                )
            except ScenarioTimeout as error:
                self._stop_pool(error.worker_pid)
                self.skipped_scenarios.add(error.scenario_index)
                self.on_timeout(error)
                progress_bar.update(1)
                # _start_pool creates a new queue; stale events from the killed
                # attempt cannot be attributed to its replacement workers.

    def __exit__(self, *args: Any) -> None:
        """Stop all workers and the Manager on success, error, or interruption."""
        try:
            self._stop_pool()
        finally:
            if self.manager is not None:
                self.manager.shutdown()
