"""
Main dynamic data generation pipeline.

Entry point: generate_dynamic_data(config_path)

Analogous to generate_power_flow_data in generate.py but extended with a
dynamic simulation step. Produces both static PF snapshots (Parquet) and
dynamic time-series (Zarr).

All outputs live under a single root, ``settings.data_dir`` — there is no separate
dynamic output directory. The layout reuses the static pipeline's base_path so a
run is one self-contained tree::

    {settings.data_dir}/{network.name}/raw/
        args.log, error.log, scenarios_*.{parquet,html,log}
        solver_log/                    (only when settings.enable_solver_logs)
        dynamic/
            bus_data.parquet, branch_data.parquet, gen_data.parquet,
            y_bus_data.parquet, runtime_data.parquet
            final_state_values.parquet   (only when FinalStateValue rows are monitored)
            dynamic_results.zarr/
            reports/
            metadata.json

Outputs are written incrementally, one large chunk at a time, so peak memory
tracks ``settings.large_chunk_size`` rather than the size of the whole dataset.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import yaml

from gridfm_datakit.dynamic import load_raw_inputs
from gridfm_datakit.generate import _prepare_network_and_scenarios, _setup_environment
from gridfm_datakit.process.solver_output import SolverVerbosity
from gridfm_datakit.utils.column_names import (
    BRANCH_COLUMNS,
    BUS_COLUMNS,
    GEN_COLUMNS,
    RUNTIME_COLUMNS,
    YBUS_COLUMNS,
)
from gridfm_datakit.utils.param_handler import NestedNamespace

# Progress/status logger for the dynamic pipeline. A NullHandler keeps the
# library quiet unless the application (or _configure_logging below, driven by
# dynamic.logging.verbosity) attaches a handler.
logger = logging.getLogger("gridfm_datakit.dynamic")
logger.addHandler(logging.NullHandler())

# dynamic.logging.verbosity -> progress-logger threshold. Mirrors SolverVerbosity
# so the dynamic knob shares one vocabulary with the solver-output policy.
_VERBOSITY_TO_LEVEL = {
    SolverVerbosity.SILENT: logging.ERROR,
    SolverVerbosity.ERROR: logging.ERROR,
    SolverVerbosity.WARNING: logging.WARNING,
    SolverVerbosity.INFO: logging.INFO,
    SolverVerbosity.DEBUG: logging.DEBUG,
}


def _configure_logging(config: NestedNamespace) -> None:
    """Set the dynamic logger level/handler from dynamic.logging.verbosity.

    Idempotent: attaches at most one StreamHandler so the summary stays visible
    by default (verbosity "info") without duplicating on repeated calls.
    """
    logging_cfg = getattr(getattr(config, "dynamic", None), "logging", None)
    verbosity = getattr(logging_cfg, "verbosity", "info") if logging_cfg else "info"
    try:
        level = _VERBOSITY_TO_LEVEL[SolverVerbosity.parse(verbosity)]
    except ValueError:
        level = logging.INFO
    logger.setLevel(level)
    # Only attach our own handler when nothing else will emit these records —
    # neither a prior call here nor an application that configured the root
    # logger (e.g. logging.basicConfig). Otherwise records would print twice
    # (once via our handler, once via the propagated root handler).
    root_configured = bool(logging.getLogger().handlers)
    has_own_handler = any(
        not isinstance(h, logging.NullHandler) for h in logger.handlers
    )
    if not root_configured and not has_own_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[dynamic] %(message)s"))
        logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_dynamic_data(
    config: Union[str, Dict[str, Any], NestedNamespace],
) -> Dict[str, str]:
    """Generate dynamic simulation data from a YAML config.
    Accepted format includes: a string for the path, a dictionnary or a NestedNamespace.

    Runs the full pipeline:
    1. Validate config.
    2. Prepare network + load scenarios.
    3. Load and prepare Dynawo mappings.
    4. Build solver parameters.
    5. Run distributed dynamic simulations.
    6. Save static (Parquet) + dynamic (Zarr) outputs.

    Args
    ----
    config : str | dict | NestedNamespace
        Path to a YAML config file, a plain dict, or a NestedNamespace.

    Returns
    -------
    dict
        Paths to all generated artifacts, all rooted at ``settings.data_dir``:
        the static keys (``bus_data``, ``branch_data``, ``gen_data``,
        ``y_bus_data``, ``runtime_data``, ``error_log``, ``args_log``,
        ``solver_log_dir``, ``scenarios``) plus ``dynamic_results`` (Zarr store),
        ``dynamic_reports_dir`` and ``metadata``. ``final_state_values`` is
        present only when the variables table declares FinalStateValue rows.

    Raises
    ------
    ValueError
        If ``network.reader != "powsybl"``, ``dynamic.dynamic_solver`` is not set,
        or the removed ``dynamic.output_dir`` key is still present.
    """

    # --- Step 0: load and validate config ---
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    if isinstance(config, dict):
        args = NestedNamespace(**config)
    else:
        args = config

    _validate_dynamic_config(args)
    _configure_logging(args)

    # --- Step 1: standard environment setup (reuse generate.py logic) ---
    args, base_path, file_paths, seed = _setup_environment(args)
    # _setup_environment derives solver_log_dir (honouring enable_solver_logs)
    # into file_paths; publish it on settings so the distributed dynamic loop
    # (which reads config.settings.solver_log_dir) routes OPF + Dynawo native
    # output to files instead of dropping it.
    args.settings.solver_log_dir = file_paths["solver_log_dir"]

    # The dynamic pipeline reports progress per chunk through the
    # "gridfm_datakit.dynamic" logger, not tqdm, so _setup_environment's tqdm.log
    # stays empty and is not exported.
    #
    # It is only deleted when settings.overwrite is set: base_path has then just
    # been wiped and recreated, so the file there is certainly ours. Otherwise
    # base_path may be shared with an earlier *static* run whose accumulated
    # tqdm.log must be left alone.
    tqdm_log = file_paths.pop("tqdm_log", None)
    if tqdm_log is not None and getattr(args.settings, "overwrite", False):
        Path(tqdm_log).unlink(missing_ok=True)

    # --- Step 2: network + scenarios (reuse generate.py logic) ---
    # Only the scenarios and meta["network_path"] are used downstream: workers
    # reload the network themselves from that path (see _process_dynamic_chunk).
    _, scenarios, meta = _prepare_network_and_scenarios(args, file_paths, seed)

    # --- Step 3: dynamic inputs ---
    dynamic_inputs = load_raw_inputs(args)

    # --- Step 4: output directory ---
    # Single root: everything this run produces lives under settings.data_dir, in
    # the same base_path (data_dir/<network>/raw) the static pipeline uses for its
    # logs and scenarios. The dynamic artifacts go one level down, in dynamic/,
    # because the static pipeline writes bus_data.parquet as a *partitioned
    # directory* while we write it as a flat file — same name, different kind, so
    # they must not share a directory.
    #
    # The writer owns this directory: it recreates it from scratch, so a re-run can
    # never mix fresh artifacts with a previous run's leftovers.
    dynamic_solver = args.dynamic.dynamic_solver
    output_dir = Path(base_path) / "dynamic"

    # --- Steps 5 & 6: simulate and save, one chunk at a time ---
    # Each chunk is written and released before the next runs, so peak memory
    # tracks settings.large_chunk_size rather than the whole dataset. Dynamic
    # curves are far larger than static snapshots, which is why this streams
    # instead of collecting every sample first.
    from gridfm_datakit.dynamic.process_dynamic import iter_dynamic_simulations

    writer = _DynamicDataWriter(output_dir, file_paths, args, seed)
    for chunk_results in iter_dynamic_simulations(
        network_path=meta["network_path"],
        scenarios=scenarios,
        dynamic_inputs=dynamic_inputs,
        dynamic_solver=dynamic_solver,
        config=args,
        error_log_file=file_paths["error_log"],
        seed=seed,
    ):
        writer.write_chunk(chunk_results)
        del chunk_results
        gc.collect()
    writer.close()

    return file_paths


def _validate_dynamic_config(args: NestedNamespace) -> None:
    """Raise ValueError for config issues that would cause silent failures."""

    reader = getattr(args.network, "reader", None)
    if reader != "powsybl":
        raise ValueError(
            f"Dynamic simulations require network.reader='powsybl', "
            f"got {reader!r}. Set 'reader: powsybl' in the network block.",
        )

    dyn = getattr(args, "dynamic", None)
    if dyn is None:
        raise ValueError(
            "Config is missing the 'dynamic:' block. "
            "Add a dynamic: section with at least dynamic_solver: 'dynawo'.",
        )

    dynamic_solver = getattr(dyn, "dynamic_solver", None)
    if not dynamic_solver:
        raise ValueError(
            "Config is missing dynamic.dynamic_solver. "
            "Set 'dynamic_solver: dynawo' in the dynamic block.",
        )

    # dynamic.output_dir used to be a second, independent output root, which split
    # a single run's artifacts across two unrelated trees. Outputs are now rooted
    # at settings.data_dir like the static pipeline. Fail loudly rather than
    # silently ignore the key and write somewhere the user does not expect.
    if getattr(dyn, "output_dir", None) is not None:
        raise ValueError(
            "dynamic.output_dir has been removed: dynamic outputs are now written "
            "under settings.data_dir, in {data_dir}/{network.name}/raw/dynamic/. "
            "Delete 'output_dir' from the dynamic block and set settings.data_dir "
            "instead.",
        )

    # Without at least one scenario there is nothing to chunk: np.array_split would
    # raise "number sections must be larger than 0" from inside numpy.
    n_scenarios = getattr(getattr(args, "load", None), "scenarios", 0)
    if not n_scenarios or n_scenarios < 1:
        raise ValueError(
            f"load.scenarios must be >= 1, got {n_scenarios!r}. "
            "Set the number of load scenarios to generate.",
        )

    # Fail fast on a missing Dynawo installation. pypowsybl.dynamic imports fine
    # without it, so otherwise the run only dies once the workers reach
    # Simulation.run(), with an opaque provider-instantiation error.
    if dynamic_solver == "dynawo":
        from gridfm_datakit.dynamic.dynawo.api import check_dynawo_available

        check_dynawo_available()


def _time_axis_seconds(curves: pd.DataFrame) -> np.ndarray:
    """Return the curves' time axis as float seconds of simulation time.

    pypowsybl exposes the curves index as pandas Timestamps built from the raw
    Dynawo time (so t=0 s is the epoch). Convert back to seconds, which is what
    the config speaks in (``solver_parameters.start_time`` / ``stop_time``).
    Falls back to a plain float cast if the index is already numeric.
    """
    index = curves.index
    if isinstance(index, pd.DatetimeIndex):
        return index.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9
    return np.asarray(index, dtype="float64")


def _final_state_values_to_mapping(fsv: Any) -> Dict[str, float]:
    """Normalise a solver's final-state-value frame to a {name: value} mapping.

    pypowsybl returns a DataFrame indexed by the flattened variable name
    (``<model_id>_<variable>``) with a single "values" column. The two-column
    layout its docstring describes is handled too, so a change on that side
    degrades to a wrong column rather than a crash. Order is never relied on:
    Dynawo does not return the variables in the order they were registered.
    """
    if fsv is None or len(fsv) == 0:
        return {}
    if fsv.shape[1] == 1:
        names, values = fsv.index, fsv.iloc[:, 0]
    else:
        names, values = fsv.iloc[:, 0], fsv.iloc[:, 1]
    return {str(name): float(value) for name, value in zip(names, values)}


class _DynamicDataWriter:
    """Incremental writer for one dynamic run's outputs.

    Chunks are written and released as they complete, so peak memory tracks the
    largest chunk rather than the whole dataset — matching the static pipeline,
    which saves per large chunk and drops the batch (see generate._save_generated_data).
    ``settings.large_chunk_size`` is what bounds memory here.

    Owns ``output_dir`` ({data_dir}/{network}/raw/dynamic): it is recreated from
    scratch, so a re-run can never mix fresh artifacts with a previous run's
    leftovers. Nothing outside it is touched.

    The directory is only created on the first chunk that actually carries a
    sample. A run in which every scenario fails therefore leaves the previous
    run's data untouched, and ``close()`` raises instead of returning paths to
    artifacts that were never written.

    Layout under output_dir:
      bus_data.parquet
      branch_data.parquet
      gen_data.parquet
      y_bus_data.parquet
      runtime_data.parquet
      final_state_values.parquet  ← only when the run monitors FinalStateValue rows
      dynamic_results.zarr/       ← shape (n_samples, n_variables, n_timesteps)
      reports/                    ← one Dynawo report per sample
      metadata.json
    """

    _STATIC_TABLES = (
        ("bus_data", "bus", BUS_COLUMNS),
        ("branch_data", "branch", BRANCH_COLUMNS),
        ("gen_data", "gen", GEN_COLUMNS),
        ("y_bus_data", "Y_bus", YBUS_COLUMNS),
        ("runtime_data", "runtime", RUNTIME_COLUMNS),
    )

    def __init__(
        self,
        output_dir: Path,
        file_paths: Dict[str, str],
        config: NestedNamespace,
        seed: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.file_paths = file_paths
        self.config = config
        self.seed = seed

        self._started = False
        self._parquet_writers: Dict[str, Any] = {}
        self._store = None
        self._curves = None
        self._time = None

        # Metadata accumulated across chunks; nothing else survives a chunk.
        self.n_samples = 0
        self.n_variables = 0
        self.max_n_timesteps = 0
        self.timesteps_per_scenario: List[int] = []
        self.static_keys: List[tuple] = []
        self.dynamic_scenarios: List[int] = []
        self.dynamic_perturbations: List[int] = []
        self.variable_names: List[str] = []
        self.fsv_names: List[str] = []
        self.report_index: List[str] = []

        logging_cfg = getattr(getattr(config, "dynamic", None), "logging", None)
        self.save_reports = (
            getattr(logging_cfg, "save_reports", True) if logging_cfg else True
        )

    # -- lifecycle ---------------------------------------------------------

    def _start(self) -> None:
        """Create output_dir on first use. Safe to wipe: it holds only our own
        artifacts — never inputs, logs or scenarios, which live one level up."""
        if self._started:
            return
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._started = True

    def write_chunk(self, results: List[Dict[str, Any]]) -> None:
        """Append one chunk of samples, then let the caller drop them."""
        if not results:
            return
        self._start()
        self._write_static(results)
        self._write_curves(results)
        self._write_reports(results)
        self.n_samples += len(results)

    def close(self) -> None:
        """Flush every writer and emit metadata.json.

        Raises
        ------
        RuntimeError
            If no sample was ever written, i.e. every scenario failed.
        """
        for writer in self._parquet_writers.values():
            writer.close()
        self._parquet_writers.clear()

        if not self._started:
            raise RuntimeError(
                "Dynamic generation produced no samples: every scenario failed. "
                "See the error log for per-scenario causes. Nothing was written to "
                f"{self.output_dir}; any files already there are from a previous run.",
            )

        self._write_metadata()

    # -- static snapshot ---------------------------------------------------

    def _write_static(self, results: List[Dict[str, Any]]) -> None:
        """Append the static PF snapshot rows for this chunk.

        Every element row is tagged with its (scenario_index, perturbation_index)
        so the static snapshot (features / initial conditions) can be joined back
        to the dynamic trajectory (labels) by key rather than by row position.
        This keeps the two modalities aligned even when a sample contributes to
        only one.
        """
        keys, rows = [], {key: [] for key, _, _ in self._STATIC_TABLES}
        for result in results:
            pf_data = result["pf_data"]
            if pf_data is None:
                continue
            keys.append(
                (result["scenario_index"], result.get("perturbation_index", 0)),
            )
            for key, pf_key, _ in self._STATIC_TABLES:
                rows[key].append(pf_data[pf_key])
        if not keys:
            return
        self.static_keys.extend(keys)

        for key, _, columns in self._STATIC_TABLES:
            frames = []
            for (scenario_id, perturbation_id), array in zip(keys, rows[key]):
                array = np.atleast_2d(array)
                frame = pd.DataFrame(array, columns=columns[: array.shape[1]])
                frame.insert(0, "perturbation_index", perturbation_id)
                frame.insert(0, "scenario_index", scenario_id)
                frames.append(frame)
            self._append_parquet(key, pd.concat(frames, ignore_index=True))

        self._write_final_state_values(results)

    def _write_final_state_values(self, results: List[Dict[str, Any]]) -> None:
        """Append one row per sample of FinalStateValue results, if any.

        These are per-sample scalars, not trajectories, so they belong in a table
        keyed like the rest of the static snapshot rather than in the Zarr store.
        The file is only created when the run actually monitors FinalStateValue
        rows; a curves-only run produces no such table.

        The column set is fixed by the first chunk that carries values. A later
        sample reporting a different set would silently change the schema, so the
        frame is reindexed onto the known columns: unexpected names are dropped
        and missing ones become NaN.
        """
        rows = []
        for result in results:
            dynamic_results = result.get("dynamic_results")
            values = _final_state_values_to_mapping(
                getattr(dynamic_results, "final_state_values", None),
            )
            if not values:
                continue
            if not self.fsv_names:
                self.fsv_names = sorted(values)
            row = {
                "scenario_index": result["scenario_index"],
                "perturbation_index": result.get("perturbation_index", 0),
            }
            row.update(values)
            rows.append(row)
        if not rows:
            return
        frame = pd.DataFrame(rows).reindex(
            columns=["scenario_index", "perturbation_index"] + self.fsv_names,
        )
        self._append_parquet("final_state_values", frame)

    def _append_parquet(self, key: str, frame: pd.DataFrame) -> None:
        """Append a row group to ``key``'s Parquet file, opening it on first use.

        The schema is fixed by the first chunk; later chunks are cast onto it so a
        column that happens to be all-integer in one chunk and float in the next
        cannot make the file unreadable.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = self._parquet_writers.get(key)
        if writer is None:
            path = str(self.output_dir / f"{key}.parquet")
            table = pa.Table.from_pandas(frame, preserve_index=False)
            writer = pq.ParquetWriter(path, table.schema)
            self._parquet_writers[key] = writer
            self.file_paths[key] = path
        else:
            table = pa.Table.from_pandas(
                frame,
                schema=writer.schema,
                preserve_index=False,
            )
        writer.write_table(table)

    # -- curves ------------------------------------------------------------

    def _write_curves(self, results: List[Dict[str, Any]]) -> None:
        """Append this chunk's trajectories to the Zarr store.

        Collected as (n_variables, n_timesteps) — the per-sample shape declared by
        the DynamicResults contract (and architecture §8, where the store is
        n_samples x n_variables x n_timesteps). Dynawo returns curves as a
        (n_timesteps, n_variables) DataFrame, so transpose here.
        """
        arrays, times, scenarios, perturbations = [], [], [], []
        for result in results:
            dynamic_results = result.get("dynamic_results")
            if dynamic_results is None or dynamic_results.dynamic_results is None:
                continue
            curves = dynamic_results.dynamic_results
            arrays.append(np.asarray(curves, dtype="float64").T)
            times.append(_time_axis_seconds(curves))
            scenarios.append(result["scenario_index"])
            perturbations.append(result.get("perturbation_index", 0))
            if not self.variable_names:
                self.variable_names = list(curves.columns)
        if not arrays:
            return

        n_variables = arrays[0].shape[0]
        # Every sample must monitor the same variables, else axis 1 of the store
        # is meaningless. Fail loudly rather than emit a corrupt array (an
        # unchecked mismatch surfaces as an opaque broadcast/zero-division error).
        seen = {array.shape[0] for array in arrays} | (
            {self.n_variables} if self._curves is not None else set()
        )
        if len(seen - {n_variables}) > 0:
            raise ValueError(
                f"Dynamic samples disagree on the number of variables: found "
                f"{sorted(seen)}. All scenarios must monitor the same variables "
                "to share one Zarr store.",
            )
        # Defence in depth. load_raw_inputs already rejects a variables table with
        # no "Curve" row, but any other route to empty curves would reach zarr as a
        # zero-width array and surface as an opaque ZeroDivisionError.
        if n_variables == 0:
            raise ValueError(
                "Dynamic simulations produced no monitored variables (empty curves). "
                "Check the 'Curve' rows of the variables input table.",
            )

        chunk_max_timesteps = max(array.shape[1] for array in arrays)
        self._ensure_curves_store(n_variables, chunk_max_timesteps)

        # Solvers with adaptive time-stepping (e.g. IDA) or unstable trajectories
        # can emit a different number of timesteps per sample. The store is sized
        # to the longest run seen so far and grown when a longer one arrives;
        # shorter samples keep the array's NaN fill. The valid length per sample is
        # recorded in metadata (timesteps_per_scenario) so consumers can mask it.
        if chunk_max_timesteps > self.max_n_timesteps:
            self.max_n_timesteps = chunk_max_timesteps
            self._curves.resize(
                (self._curves.shape[0], n_variables, self.max_n_timesteps),
            )
            self._time.resize((self._time.shape[0], self.max_n_timesteps))

        start = self._curves.shape[0]
        self._curves.resize(
            (start + len(arrays), n_variables, self.max_n_timesteps),
        )
        self._time.resize((start + len(arrays), self.max_n_timesteps))
        for offset, array in enumerate(arrays):
            n_timesteps = array.shape[1]
            self._curves[start + offset, :, :n_timesteps] = array
            self._time[start + offset, : times[offset].shape[0]] = times[offset]
            self.timesteps_per_scenario.append(n_timesteps)

        self.dynamic_scenarios.extend(scenarios)
        self.dynamic_perturbations.extend(perturbations)

    def _ensure_curves_store(self, n_variables: int, n_timesteps: int) -> None:
        """Create the Zarr store on first use, empty along the sample axis.

        ``fill_value=nan`` is what makes the padding free: a resize along either
        the sample or the time axis exposes NaN, so a short run needs no explicit
        padding pass.
        """
        if self._curves is not None:
            return
        import zarr

        self.n_variables = n_variables
        self.max_n_timesteps = n_timesteps
        path = str(self.output_dir / "dynamic_results.zarr")
        self._store = zarr.open(path, mode="w")
        self.file_paths["dynamic_results"] = path

        shape = (0, n_variables, n_timesteps)
        chunks = (1, n_variables, n_timesteps)
        # The Zarr array-creation API differs between v2 and v3; support both so
        # the pipeline works regardless of which major version is installed.
        if hasattr(self._store, "create_array"):  # zarr v3
            self._curves = self._store.create_array(
                "curves",
                shape=shape,
                dtype="float64",
                chunks=chunks,
                fill_value=np.nan,
                compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            )
            self._time = self._store.create_array(
                "time",
                shape=(0, n_timesteps),
                dtype="float64",
                fill_value=np.nan,
            )
        else:  # zarr v2
            import numcodecs

            self._curves = self._store.create_dataset(
                "curves",
                shape=shape,
                dtype="float64",
                chunks=chunks,
                fill_value=np.nan,
                compressor=numcodecs.Blosc(cname="zstd", clevel=3),
            )
            self._time = self._store.create_dataset(
                "time",
                shape=(0, n_timesteps),
                dtype="float64",
                fill_value=np.nan,
            )

    # -- reports -----------------------------------------------------------

    def _write_reports(self, results: List[Dict[str, Any]]) -> None:
        """Persist each sample's Dynawo report.

        The pypowsybl ReportNode (model build-up + convergence) is a JSON string on
        DynamicResults.report, and the documented way to diagnose a failed or
        degenerate run. Controlled by dynamic.logging.save_reports (default True).
        """
        if not self.save_reports:
            return
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        self.file_paths["dynamic_reports_dir"] = str(reports_dir)
        for result in results:
            dynamic_results = result.get("dynamic_results")
            if (
                dynamic_results is None
                or getattr(dynamic_results, "report", None) is None
            ):
                continue
            name = (
                f"scenario_{result['scenario_index']}"
                f"_perturbation_{result.get('perturbation_index', 0)}.json"
            )
            (reports_dir / name).write_text(str(dynamic_results.report))
            self.report_index.append(name)

    # -- metadata ----------------------------------------------------------

    def _write_metadata(self) -> None:
        config_hash = hashlib.md5(
            json.dumps(self.config.to_dict(), sort_keys=True, default=str).encode(),
        ).hexdigest()

        metadata = {
            "generated_at": datetime.now().isoformat(),
            "seed": self.seed,
            # SAMPLES, not load scenarios: a topology perturbation expands one load
            # scenario into several samples, so this exceeds config.load.scenarios
            # whenever topology_perturbation is enabled. It is the length of axis 0
            # of curves. Never compare it against load.scenarios.
            "n_samples": self.n_samples,
            "n_samples_with_curves": len(self.dynamic_scenarios),
            "variable_names": self.variable_names,
            "n_variables": self.n_variables,
            # Store dimensions: curves is (n_samples, n_variables, n_timesteps),
            # NaN-padded to n_timesteps; timesteps_per_scenario gives each run's
            # valid (unpadded) length.
            "n_timesteps": self.max_n_timesteps,
            "timesteps_per_scenario": self.timesteps_per_scenario,
            # The simulation time (in seconds) of each column of curves lives in the
            # store's "time" array, shaped (n_samples, n_timesteps) and NaN-padded to
            # match. Per-sample, since an adaptive-step solver gives each run its own
            # time grid. Never assume a shared/uniform time axis — read this instead.
            "time_units": "seconds",
            # Join keys: the Parquet rows are labelled by (scenario_index,
            # perturbation_index) columns; the Zarr curves slices by the matching
            # "scenario_index"/"perturbation_index" coordinate arrays. Join the two
            # modalities on this key pair, never on row/slice position.
            "static_scenario_index": [key[0] for key in self.static_keys],
            "static_perturbation_index": [key[1] for key in self.static_keys],
            "dynamic_scenario_index": self.dynamic_scenarios,
            "dynamic_perturbation_index": self.dynamic_perturbations,
            # Columns of final_state_values.parquet, in order. Empty when the run
            # monitors no FinalStateValue rows, in which case no such file exists.
            "final_state_value_names": self.fsv_names,
            # Per-sample Dynawo solver reports under reports/ (empty if disabled).
            "reports": self.report_index,
            "config_hash": config_hash,
        }

        if self._store is not None:
            self._write_coordinates()

        path = str(self.output_dir / "metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        self.file_paths["metadata"] = path

        logger.info(
            "Saved %d samples to %s (%d with dynamic results, %d reports).",
            self.n_samples,
            self.output_dir,
            len(self.dynamic_scenarios),
            len(self.report_index),
        )

    def _write_coordinates(self) -> None:
        """Write the (scenario_index, perturbation_index) coordinate arrays.

        They map each curves slice (axis 0) back to the sample it came from, so the
        Zarr labels join to the Parquet snapshot by key. Written once at close,
        when the final sample count is known.
        """
        n_samples = len(self.dynamic_scenarios)
        for name, values in (
            ("scenario_index", self.dynamic_scenarios),
            ("perturbation_index", self.dynamic_perturbations),
        ):
            data = np.asarray(values, dtype="int64")
            if hasattr(self._store, "create_array"):  # zarr v3
                array = self._store.create_array(
                    name,
                    shape=(n_samples,),
                    dtype="int64",
                )
                array[:] = data
            else:  # zarr v2
                self._store.create_dataset(
                    name,
                    data=data,
                    shape=(n_samples,),
                    dtype="int64",
                )


def _save_generated_data(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    file_paths: Dict[str, str],
    config: NestedNamespace,
    seed: int,
) -> None:
    """Write a complete result list in one shot.

    Thin wrapper over :class:`_DynamicDataWriter` for callers that already hold
    every sample in memory (tests, ad-hoc scripts). The pipeline itself streams
    chunk by chunk instead — see generate_dynamic_data.
    """
    writer = _DynamicDataWriter(output_dir, file_paths, config, seed)
    writer.write_chunk(all_results)
    writer.close()
