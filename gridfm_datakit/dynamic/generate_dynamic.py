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
            dynamic_results.zarr/
            reports/
            metadata.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from gridfm_datakit.dynamic import DynamicResults, load_raw_inputs
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
        ``dynamic_reports_dir`` and ``metadata``.

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

    # The dynamic pipeline reports progress through the "gridfm_datakit.dynamic"
    # logger (per-chunk), not tqdm — so _setup_environment's tqdm.log would only
    # ever be an empty file. Stop exporting it.
    #
    # Only remove the file when settings.overwrite is set, i.e. when
    # _setup_environment has just wiped and recreated base_path so the tqdm.log
    # there is certainly ours. Otherwise base_path may be reused from an earlier
    # *static* run, whose accumulated tqdm.log we must not destroy.
    tqdm_log = file_paths.pop("tqdm_log", None)
    if tqdm_log is not None and getattr(args.settings, "overwrite", False):
        Path(tqdm_log).unlink(missing_ok=True)

    # --- Step 2: network + scenarios (reuse generate.py logic) ---
    # TODO: discuss with YE: just a function to prep load scenarios and the path
    # we don't need the net, since we'll pass the path instead of the net along the pipeline
    net, scenarios, meta = _prepare_network_and_scenarios(args, file_paths, seed)

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
    # _save_generated_data owns this directory: it recreates it from scratch, so a
    # re-run can never mix fresh artifacts with a previous run's leftovers.
    dynamic_solver = args.dynamic.dynamic_solver
    output_dir = Path(base_path) / "dynamic"

    # --- Step 5: distributed simulation ---
    from gridfm_datakit.dynamic.process_dynamic import process_dynamic_simulations

    all_results = process_dynamic_simulations(
        network_path=meta["network_path"],
        scenarios=scenarios,
        dynamic_inputs=dynamic_inputs,
        dynamic_solver=dynamic_solver,
        config=args,
        error_log_file=file_paths["error_log"],
        seed=seed,
    )

    # --- Step 6: save outputs ---
    _save_generated_data(
        all_results=all_results,
        output_dir=output_dir,
        file_paths=file_paths,
        config=args,
        seed=seed,
    )

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

    # Fail fast on a missing Dynawo installation. pypowsybl.dynamic imports fine
    # without it, so otherwise the run only dies once the workers reach
    # Simulation.run(), with an opaque provider-instantiation error.
    if dynamic_solver == "dynawo":
        from gridfm_datakit.dynamic.dynawo.api import check_dynawo_available

        check_dynawo_available()

    # Ensure reader is set to powsybl (required for pp_net in meta)
    if getattr(args.network, "reader", "native") != "powsybl":
        args.network.reader = "powsybl"


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


def _save_generated_data(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    file_paths: Dict[str, str],
    config: NestedNamespace,
    seed: int,
) -> None:
    """Save static PF snapshot (Parquet) and dynamic time-series (Zarr).

    Owns ``output_dir`` ({data_dir}/{network}/raw/dynamic): it is recreated from
    scratch on every call, so a re-run can never mix fresh artifacts with a
    previous run's leftovers. Nothing outside it is touched.

    Layout under output_dir:
      bus_data.parquet
      branch_data.parquet
      gen_data.parquet
      y_bus_data.parquet
      runtime_data.parquet
      dynamic_results.zarr/   ← shape (n_scenarios, n_variables, n_timesteps)
      reports/                ← one Dynawo report per sample
      metadata.json

    Args
    ----
    all_results : list of dicts
        Each dict has keys: "pf_data", "dynamic_results", "scenario_index".
    output_dir : Path
    file_paths : dict (updated in-place with output paths)
    config : NestedNamespace
    seed : int
    """
    import zarr

    # Every scenario failed. Bail out *before* clearing output_dir, so a failed
    # re-run leaves the previous run's data intact instead of destroying it — and
    # raise, so the caller never receives file_paths pointing at stale artifacts.
    if not all_results:
        raise RuntimeError(
            "Dynamic generation produced no samples: every scenario failed. "
            f"See the error log for per-scenario causes. Nothing was written to "
            f"{output_dir}; any files already there are from a previous run.",
        )

    # Recreate the directory. Safe to wipe: it holds only our own artifacts —
    # never inputs, logs or scenarios, which live one level up in base_path.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Static PF outputs → Parquet ----------------------------------------
    # Every element row is tagged with its (scenario_index, perturbation_index)
    # so the static snapshot (features / initial conditions) can be joined back to
    # the dynamic trajectory (labels) by key rather than by row position. This
    # keeps the two modalities aligned even when a sample contributes to only one.
    bus_rows, gen_rows, branch_rows = [], [], []
    y_bus_rows, runtime_rows = [], []
    static_keys = []  # list of (scenario_index, perturbation_index)
    for r in all_results:
        pf = r["pf_data"]
        if pf is None:
            continue
        static_keys.append((r["scenario_index"], r.get("perturbation_index", 0)))
        bus_rows.append(pf["bus"])
        gen_rows.append(pf["gen"])
        branch_rows.append(pf["branch"])
        y_bus_rows.append(pf["Y_bus"])
        runtime_rows.append(pf["runtime"])

    def _to_parquet(rows, keys, columns, path):
        if not rows:
            return
        frames = []
        for (scen_id, pert_id), arr in zip(keys, rows):
            arr = np.atleast_2d(arr)
            df = pd.DataFrame(arr, columns=columns[: arr.shape[1]])
            df.insert(0, "perturbation_index", pert_id)
            df.insert(0, "scenario_index", scen_id)
            frames.append(df)
        pd.concat(frames, ignore_index=True).to_parquet(
            path,
            index=False,
            engine="pyarrow",
        )

    bus_path = str(output_dir / "bus_data.parquet")
    branch_path = str(output_dir / "branch_data.parquet")
    gen_path = str(output_dir / "gen_data.parquet")
    y_bus_path = str(output_dir / "y_bus_data.parquet")
    runtime_path = str(output_dir / "runtime_data.parquet")

    _to_parquet(bus_rows, static_keys, BUS_COLUMNS, bus_path)
    _to_parquet(gen_rows, static_keys, GEN_COLUMNS, gen_path)
    _to_parquet(branch_rows, static_keys, BRANCH_COLUMNS, branch_path)
    # Y-bus and runtime complete the static snapshot at parity with the static
    # pipeline's _save_generated_data, which exports both. pf_data already
    # carries them, so omitting them silently dropped data.
    _to_parquet(y_bus_rows, static_keys, YBUS_COLUMNS, y_bus_path)
    _to_parquet(runtime_rows, static_keys, RUNTIME_COLUMNS, runtime_path)

    file_paths["bus_data"] = bus_path
    file_paths["branch_data"] = branch_path
    file_paths["gen_data"] = gen_path
    file_paths["y_bus_data"] = y_bus_path
    file_paths["runtime_data"] = runtime_path

    # ---- Dynamic time-series → Zarr -----------------------------------------
    # Collect per-scenario arrays as (n_variables, n_timesteps) — the per-scenario
    # shape declared by the DynamicResults contract (and architecture §8, where the
    # store is n_scenarios x n_variables x n_timesteps). Dynawo returns curves as a
    # (n_timesteps, n_variables) DataFrame, so transpose here.
    dyn_arrays = []
    time_arrays: List[np.ndarray] = []
    timesteps_per_scenario: List[int] = []
    dynamic_scenarios: List[int] = []
    dynamic_perturbations: List[int] = []
    for r in all_results:
        dr: Optional[DynamicResults] = r.get("dynamic_results")
        if dr is None or dr.dynamic_results is None:
            continue
        arr = np.asarray(
            dr.dynamic_results,
            dtype="float64",
        ).T  # (n_variables, n_timesteps)
        dyn_arrays.append(arr)
        time_arrays.append(_time_axis_seconds(dr.dynamic_results))
        timesteps_per_scenario.append(arr.shape[1])
        dynamic_scenarios.append(r["scenario_index"])
        dynamic_perturbations.append(r.get("perturbation_index", 0))

    zarr_path = str(output_dir / "dynamic_results.zarr")
    n_variables = 0
    max_n_timesteps = 0
    if dyn_arrays:
        n_scenarios = len(dyn_arrays)
        n_variables = dyn_arrays[0].shape[0]
        # Every sample must monitor the same variables, else axis 1 of the store
        # is meaningless. Fail loudly rather than emit a corrupt array (an
        # unchecked mismatch surfaces as an opaque broadcast/zero-division error).
        mismatched = {a.shape[0] for a in dyn_arrays} - {n_variables}
        if mismatched:
            raise ValueError(
                f"Dynamic samples disagree on the number of variables: found "
                f"{sorted(mismatched | {n_variables})}. All scenarios must monitor "
                f"the same variables to share one Zarr store.",
            )
        # Solvers with adaptive time-stepping (e.g. IDA) or unstable trajectories
        # can emit a different number of timesteps per scenario. Size the store to
        # the longest run and NaN-pad shorter scenarios along the time axis; the
        # valid length per scenario is recorded in metadata (timesteps_per_scenario)
        # so consumers can mask the padding.
        max_n_timesteps = max(a.shape[1] for a in dyn_arrays)
        shape = (n_scenarios, n_variables, max_n_timesteps)
        chunks = (1, n_variables, max_n_timesteps)
        store = zarr.open(zarr_path, mode="w")
        # The Zarr array-creation API differs between v2 and v3; support both so
        # the pipeline works regardless of which major version is installed.
        if hasattr(store, "create_array"):  # zarr v3
            z = store.create_array(
                "curves",
                shape=shape,
                dtype="float64",
                chunks=chunks,
                compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            )
        else:  # zarr v2
            import numcodecs

            z = store.create_dataset(
                "curves",
                shape=shape,
                dtype="float64",
                chunks=chunks,
                compressor=numcodecs.Blosc(cname="zstd", clevel=3),
            )
        for i, arr in enumerate(dyn_arrays):
            n_t = arr.shape[1]
            if n_t == max_n_timesteps:
                z[i] = arr
            else:
                padded = np.full(
                    (n_variables, max_n_timesteps),
                    np.nan,
                    dtype="float64",
                )
                padded[:, :n_t] = arr
                z[i] = padded

        # (scenario_index, perturbation_index) coordinates: map each curves slice
        # (axis 0) back to the sample it came from, so the Zarr labels join to the
        # Parquet snapshot by key.
        def _write_coord(name, values):
            data = np.asarray(values, dtype="int64")
            if hasattr(store, "create_array"):  # zarr v3
                arr = store.create_array(name, shape=(n_scenarios,), dtype="int64")
                arr[:] = data
            else:  # zarr v2
                store.create_dataset(
                    name,
                    data=data,
                    shape=(n_scenarios,),
                    dtype="int64",
                )

        _write_coord("scenario_index", dynamic_scenarios)
        _write_coord("perturbation_index", dynamic_perturbations)

        # "time" coordinate: the simulation time (seconds) of every column of
        # curves, shaped (n_scenarios, n_timesteps) and NaN-padded like the curves.
        #
        # It is stored per scenario, not once for the whole store, because with an
        # adaptive-step solver (IDA) each run picks its own time grid: column t then
        # means a *different* physical instant in different scenarios, and without
        # this array the mapping back to time is unrecoverable. With a fixed-step
        # solver (SIM) every row is identical and this is merely redundant — cheap
        # insurance rather than a 4-D tensor.
        time_grid = np.full((n_scenarios, max_n_timesteps), np.nan, dtype="float64")
        for i, times in enumerate(time_arrays):
            time_grid[i, : times.shape[0]] = times
        if hasattr(store, "create_array"):  # zarr v3
            t = store.create_array(
                "time",
                shape=time_grid.shape,
                dtype="float64",
            )
            t[:] = time_grid
        else:  # zarr v2
            store.create_dataset(
                "time",
                data=time_grid,
                shape=time_grid.shape,
                dtype="float64",
            )

    file_paths["dynamic_results"] = zarr_path

    # ---- Dynawo solver reports → JSON ---------------------------------------
    # Each simulation's pypowsybl ReportNode (model build-up + convergence) is a
    # JSON string on DynamicResults.report. It is the documented way to diagnose
    # a failed/degenerate run, so persist one file per sample, keyed like every
    # other output. Controlled by dynamic.logging.save_reports (default True).
    save_reports = getattr(getattr(config, "dynamic", None), "logging", None)
    save_reports = getattr(save_reports, "save_reports", True) if save_reports else True
    report_index: List[str] = []
    if save_reports:
        # No stale-report cleanup needed: output_dir was recreated from scratch above.
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for r in all_results:
            dr = r.get("dynamic_results")
            if dr is None or getattr(dr, "report", None) is None:
                continue
            name = (
                f"scenario_{r['scenario_index']}"
                f"_perturbation_{r.get('perturbation_index', 0)}.json"
            )
            (reports_dir / name).write_text(str(dr.report))
            report_index.append(name)
        file_paths["dynamic_reports_dir"] = str(reports_dir)

    # ---- metadata.json -------------------------------------------------------
    # Determine variable names from the first successful result
    variable_names: List[str] = []
    for r in all_results:
        dr = r.get("dynamic_results")
        if dr and dr.dynamic_results is not None:
            variable_names = list(dr.dynamic_results.columns)
            break

    config_hash = hashlib.md5(
        json.dumps(config.to_dict(), sort_keys=True, default=str).encode(),
    ).hexdigest()

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "seed": seed,
        "n_scenarios": len(all_results),
        "n_successful": len(dyn_arrays) if dyn_arrays else 0,
        "variable_names": variable_names,
        "n_variables": n_variables,
        # Store dimensions: curves is (n_scenarios, n_variables, n_timesteps),
        # NaN-padded to n_timesteps; timesteps_per_scenario gives each run's
        # valid (unpadded) length.
        "n_timesteps": max_n_timesteps,
        "timesteps_per_scenario": timesteps_per_scenario,
        # The simulation time (in seconds) of each column of curves lives in the
        # store's "time" array, shaped (n_scenarios, n_timesteps) and NaN-padded to
        # match. Per-scenario, since an adaptive-step solver gives each run its own
        # time grid. Never assume a shared/uniform time axis — read this instead.
        "time_units": "seconds",
        # Join keys: the Parquet rows are labelled by (scenario_index,
        # perturbation_index) columns; the Zarr curves slices by the matching
        # "scenario_index"/"perturbation_index" coordinate arrays. Join the two
        # modalities on this key pair, never on row/slice position.
        "static_scenario_index": [k[0] for k in static_keys],
        "static_perturbation_index": [k[1] for k in static_keys],
        "dynamic_scenario_index": dynamic_scenarios,
        "dynamic_perturbation_index": dynamic_perturbations,
        # Per-sample Dynawo solver reports under reports/ (empty if disabled).
        "reports": report_index,
        "config_hash": config_hash,
    }

    meta_path = str(output_dir / "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    file_paths["metadata"] = meta_path

    logger.info(
        "Saved %d samples to %s (%d with dynamic results, %d reports).",
        len(all_results),
        output_dir,
        len(dyn_arrays),
        len(report_index),
    )
