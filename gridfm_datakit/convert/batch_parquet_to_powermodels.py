"""Batch-convert parquet scenarios to PowerModels JSON."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from gridfm_datakit.convert.parquet_to_powermodels import frames_to_json
from gridfm_datakit.utils.utils import get_num_scenarios, n_scenario_per_partition

TABLES = ("bus", "gen", "branch")
SORT = {"bus": "bus", "gen": "idx", "branch": "idx"}


def convert_case(
    case_dir: str | Path,
    max_samples: int = 10_000,
    workers: int | None = None,
) -> None:
    """Read raw/ parquets, write scenario JSONs under powermodels/."""
    case_dir = Path(case_dir)
    raw_dir = case_dir / "raw"
    out_dir = case_dir / "powermodels"
    out_dir.mkdir(exist_ok=True)
    workers = workers or min(32, cpu_count())

    n = min(max_samples, get_num_scenarios(str(raw_dir)))
    last_partition = (n - 1) // n_scenario_per_partition

    for partition in tqdm(range(last_partition + 1), desc=case_dir.name, unit="partition"):
        start = partition * n_scenario_per_partition
        end = min((partition + 1) * n_scenario_per_partition, n)
        filters = [("scenario_partition", "=", partition)]
        frames = {
            name: pd.read_parquet(raw_dir / f"{name}_data.parquet", filters=filters, engine="pyarrow")
            for name in TABLES
        }
        scenarios = frames["bus"]["scenario"]
        assert scenarios.min() == start
        assert scenarios.max() + 1 == start + n_scenario_per_partition

        def convert(scenario: int) -> None:
            out = out_dir / f"scenario_{scenario}.json"
            if out.exists():
                return
            parts = []
            for name in TABLES:
                df = frames[name].loc[frames[name]["scenario"] == scenario]
                parts.append(df.sort_values(SORT[name], kind="stable").reset_index(drop=True))
            frames_to_json(*parts, str(out))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(convert, s) for s in range(start, end)]
            for future in tqdm(as_completed(futures), total=end - start, leave=False, unit="scenario"):
                future.result()
