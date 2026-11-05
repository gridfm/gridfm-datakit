"""Compare interrupted generation with an uninterrupted real-solver run."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from gridfm_datakit.cli import validate_data_directory
from integrationtests.test_generate_e2e import build_config


@pytest.mark.parametrize("mode", ["pf", "opf"])
@pytest.mark.parametrize("is_multiprocessing", [False, True], ids=["executor", "pool"])
def test_checkpoint_resume_matches_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, is_multiprocessing: bool,
) -> None:
    """Recover an uncommitted chunk and continue on the other process backend."""
    generate = importlib.import_module("gridfm_datakit.generate")
    config = build_config(
        {
            "load": {"scenarios": 8},
            "settings": {
                "mode": mode, "num_processes": 2, "large_chunk_size": 3,
                "is_multiprocessing": is_multiprocessing,
            },
        },
        str(tmp_path / "baseline"),
    )
    expected = generate.generate_power_flow_data_distributed(config)
    resumed_config = deepcopy(config)
    resumed_config["settings"].update(data_dir=str(tmp_path / "resumed"), enable_checkpointing=True)
    save = generate._save_generated_data
    writes = 0

    def interrupt(net: Any, data: list, paths: dict, root: str, args: Any) -> None:
        """Interrupt after the second save, before its checkpoint is committed."""
        nonlocal writes
        save(net, data, paths, root, args)
        writes += 1
        if writes == 2:
            raise RuntimeError("simulated interruption after parquet save")

    monkeypatch.setattr(generate, "_save_generated_data", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate.generate_power_flow_data_distributed(resumed_config)
    raw = Path(resumed_config["settings"]["data_dir"]) / config["network"]["name"] / "raw"
    state = json.loads((raw / "checkpoint.json").read_text())
    assert state["next_chunk"] == 1
    uncommitted = {path.relative_to(raw).as_posix() for path in raw.glob("*_data.parquet/**/*.parquet")} - state["files"].keys()
    assert uncommitted

    monkeypatch.setattr(generate, "_save_generated_data", save)
    resumed_config["settings"].update(overwrite=False, is_multiprocessing=not is_multiprocessing)
    actual = generate.generate_power_flow_data_distributed(resumed_config)
    assert all(not (raw / name).exists() for name in uncommitted)
    for dataset, keys in {
        "bus_data": ["scenario", "bus"],
        "gen_data": ["scenario", "idx"],
        "branch_data": ["scenario", "idx"],
        "y_bus_data": ["scenario", "index1", "index2"],
    }.items():
        expected_frame = pd.read_parquet(expected[dataset]).sort_values(keys).reset_index(drop=True)
        actual_frame = pd.read_parquet(actual[dataset]).sort_values(keys).reset_index(drop=True)
        assert_frame_equal(actual_frame, expected_frame, check_exact=True)
    assert_frame_equal(pd.read_parquet(actual["scenarios"]), pd.read_parquet(expected["scenarios"]), check_exact=True)
    assert json.loads((raw / "checkpoint.json").read_text())["next_chunk"] == 3
    assert validate_data_directory(str(raw), sn_mva=100.0, n_partitions=0, mode=mode)

    prepare = Mock(side_effect=AssertionError("completed run must not regenerate scenarios"))
    monkeypatch.setattr(generate, "_prepare_network_and_scenarios", prepare)
    generate.generate_power_flow_data_distributed(resumed_config)
    prepare.assert_not_called()
