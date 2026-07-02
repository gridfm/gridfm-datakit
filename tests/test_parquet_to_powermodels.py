"""Roundtrip test: parquet -> JSON -> Julia PF fast -> pf_post_processing."""

import os
import tempfile

import pandas as pd
import pytest

from gridfm_datakit.convert import parquet_to_json
from gridfm_datakit.convert.roundtrip_check import (
    PARQUET_TABLES,
    PF_DATA_KEYS,
    SORT_COLUMNS,
    TABLE_COLUMNS,
    configure_juliacall_env,
    define_julia_roundtrip_helpers,
    load_scenario_parquet,
    network_from_json,
)
from gridfm_datakit.process.process_network import pf_post_processing

DATA_DIR = "/dccstor/gridfm/powermodels_data/v4/finetuning/pf/case118_ieee/raw"
RUNTIME_VALUE_COLUMNS = ("ac", "dc")


def _assert_tables_equal(original: pd.DataFrame, rebuilt: pd.DataFrame, name: str) -> None:
    cols = [c for c in original.columns if c not in RUNTIME_VALUE_COLUMNS]
    pd.testing.assert_frame_equal(
        original[cols],
        rebuilt[cols],
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-9,
    )
    if name == "runtime":
        for col in RUNTIME_VALUE_COLUMNS:
            if col in original.columns:
                assert rebuilt[col].iloc[0] > 0.0


@pytest.fixture(scope="module")
def jl():
    configure_juliacall_env()
    from juliacall import Main as jl_main

    define_julia_roundtrip_helpers(jl_main, max_iter=100, tol=1e-6)
    return jl_main


@pytest.mark.skipif(not os.path.exists(DATA_DIR), reason="dataset not available")
def test_parquet_json_roundtrip_pf_fast(jl, tmp_path):
    scenario = 0
    original = load_scenario_parquet(DATA_DIR, scenario)
    load_scenario_idx = int(original["bus"]["load_scenario_idx"].iloc[0])

    json_path = os.path.join(tmp_path, f"scenario_{scenario}.json")
    parquet_to_json(DATA_DIR, scenario, json_path)

    # Smoke-test OPF on a fresh parse (benchmark entrypoint); do not mutate PF network.
    jl.solve_opf(jl.PowerModels.parse_file(json_path), 100, 1e-6)

    network = jl.PowerModels.parse_file(json_path)
    pf_result = jl.solve_pf_fast(network)
    dcpf_result = jl.solve_dcpf_fast(jl.PowerModels.parse_file(json_path))

    net = network_from_json(json_path, jl)
    pf_data = pf_post_processing(
        load_scenario_idx,
        net,
        pf_result,
        dcpf_result,
        include_dc_res=True,
    )

    for table_name in PARQUET_TABLES:
        rebuilt = pd.DataFrame(
            pf_data[PF_DATA_KEYS[table_name]],
            columns=TABLE_COLUMNS[table_name],
        ).sort_values(SORT_COLUMNS[table_name], kind="stable").reset_index(drop=True)
        _assert_tables_equal(original[table_name], rebuilt, table_name)
