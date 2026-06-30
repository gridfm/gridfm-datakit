"""
Main dynamic data generation pipeline.

Entry point: generate_dynamic_data(config_path)

Analogous to generate_power_flow_data in generate.py but extended with a
dynamic simulation step. Produces both static PF snapshots (Parquet) and
dynamic time-series (Zarr) under config.dynamic.output_dir.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from gridfm_datakit.dynamic import DynamicResults, load_raw_inputs
from gridfm_datakit.generate import _prepare_network_and_scenarios, _setup_environment
from gridfm_datakit.utils.column_names import BRANCH_COLUMNS, BUS_COLUMNS, GEN_COLUMNS
from gridfm_datakit.utils.param_handler import NestedNamespace

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_dynamic_data(
        config: Union[str, Dict[str, Any], NestedNamespace]
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
        Paths to all generated artifacts (same keys as generate_power_flow_data
        plus ``"dynamic_results_zarr"`` and ``"metadata_json"``).

    Raises
    ------
    ValueError
        If ``network.source != "powsybl"`` or ``dynamic.dynamic_solver``
        is not set.
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

    # --- Step 1: standard environment setup (reuse generate.py logic) ---
    args, base_path, file_paths, seed = _setup_environment(args)


    # --- Step 2: network + scenarios (reuse generate.py logic) ---
    # TODO: discuss with YE: just a function to prep load scenarios and the path
    # we don't need the net, since we'll pass the path instead of the net along the pipeline
    net, scenarios, meta = _prepare_network_and_scenarios(args, file_paths, seed) 

    # --- Step 3: dynamic inputs ---
    dynamic_inputs = load_raw_inputs(args)

     # --- Step 4: output directory ---
    dynamic_solver = args.dynamic.dynamic_solver
    output_dir = Path(
        getattr(args.dynamic, "output_dir", os.path.join(base_path, "dynamic")),
    )
    if output_dir.exists() and getattr(args.settings, "overwrite", False):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_paths["dynamic_output_dir"] = str(output_dir)
    file_paths["dynamic_results_zarr"] = str(output_dir / "dynamic_results.zarr")
    file_paths["metadata_json"] = str(output_dir / "metadata.json")

    # --- Step 5: distributed simulation ---
    from gridfm_datakit.dynamic.process_dynamic import process_dynamic_simulations

    all_results = process_dynamic_simulations(
        network_path=meta['network_path'],
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

    reader = getattr(args.network, 'reader', None)
    if reader != 'powsybl':
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

    # Ensure reader is set to powsybl (required for pp_net in meta)
    if getattr(args.network, "reader", "native") != "powsybl":
        args.network.reader = "powsybl"


def _save_generated_data(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    file_paths: Dict[str, str],
    config: NestedNamespace,
    seed: int,
) -> None:
    """Save static PF snapshot (Parquet) and dynamic time-series (Zarr).

    Layout under output_dir:
      bus_data.parquet
      branch_data.parquet
      gen_data.parquet
      dynamic_results.zarr/   ← shape (n_scenarios, n_timestep, n_variables)
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

    if not all_results:
        print("[dynamic] No results to save.")
        return

    # ---- Static PF outputs → Parquet ----------------------------------------
    bus_rows, gen_rows, branch_rows = [], [], []
    for r in all_results:
        pf = r["pf_data"]
        if pf is None:
            continue
        bus_rows.append(pf["bus"])
        gen_rows.append(pf["gen"])
        branch_rows.append(pf["branch"])

    def _to_parquet(rows, columns, path):
        if not rows:
            return
        arr = np.vstack(rows)
        df = pd.DataFrame(arr, columns=columns[: arr.shape[1]])
        df.to_parquet(path, index=False, engine="pyarrow")

    bus_path = str(output_dir / "bus_data.parquet")
    branch_path = str(output_dir / "branch_data.parquet")
    gen_path = str(output_dir / "gen_data.parquet")

    # TODO: this path is not the same as the path for the logs. Fix
    _to_parquet(bus_rows, BUS_COLUMNS, bus_path)
    _to_parquet(gen_rows, GEN_COLUMNS, gen_path)
    _to_parquet(branch_rows, BRANCH_COLUMNS, branch_path)

    file_paths["bus_data"] = bus_path
    file_paths["branch_data"] = branch_path
    file_paths["gen_data"] = gen_path

    # ---- Dynamic time-series → Zarr -----------------------------------------
    # Collect per-scenario arrays (n_timesteps, n_variables)
    dyn_arrays = []
    for r in all_results:
        dr: Optional[DynamicResults] = r.get("dynamic_results")
        if dr is None or dr.dynamic_results is None:
            continue
        arr = np.array(dr.dynamic_results)  # (n_timesteps, n_variables)
        dyn_arrays.append(arr)

    zarr_path = str(output_dir / "dynamic_results.zarr")
    if dyn_arrays:
        n_scenarios = len(dyn_arrays)
        n_timesteps, n_variables = dyn_arrays[0].shape
        store = zarr.open(zarr_path, mode="w")
        z = store.create_array( # create_dataset is deprecated in zarr v3
            "curves",
            shape=(n_scenarios, n_timesteps, n_variables),
            dtype="float64",
            chunks=(1, n_timesteps, n_variables),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3)
        )
        for i, arr in enumerate(dyn_arrays):
            z[i] = arr

    file_paths["dynamic_results_zarr"] = zarr_path

    # ---- metadata.json -------------------------------------------------------
    # Determine variable names from the first successful result
    variable_names: List[str] = []
    for r in all_results:
        dr = r.get("dynamic_results")
        if dr and dr.dynamic_results is not None:
            variable_names = list(dr.dynamic_results.columns)
            # variable_names = list(
            #     getattr(dr, "variable_names", [])
            #     or [f"var_{i}" for i in range(np.array(dr.dynamic_results).shape[0])],
            # )
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
        "n_timesteps": n_timesteps if dyn_arrays else 0,
        "config_hash": config_hash,
    }

    meta_path = str(output_dir / "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    file_paths["metadata_json"] = meta_path

    print(
        f"[dynamic] Saved {len(all_results)} scenarios to {output_dir} "
        f"({len(dyn_arrays)} with dynamic results).",
    )
