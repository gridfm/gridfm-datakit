"""Real-solver recovery after a deliberately stalled load scenario."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from gridfm_datakit.checkpoint import GenerationCheckpoint
from gridfm_datakit.cli import validate_data_directory
from integrationtests.test_generate_e2e import build_config


def _initialize_with_stall(*args: Any) -> None:
    """Initialize real solvers but stall load scenario 2 inside the child process."""
    from gridfm_datakit.process import process_network

    process_network._initialize_scenario_worker(*args)
    for name in ("process_scenario_pf_mode", "process_scenario_opf_mode"):
        original = getattr(process_network, name)

        def stalled(
            *positional: Any, _original: Any = original, **keywords: Any,
        ) -> list:
            """Simulate a hung solver call while preserving all other computations."""
            if positional[2] == 2:
                time.sleep(20)
            return _original(*positional, **keywords)

        setattr(process_network, name, stalled)


@pytest.mark.parametrize("mode", ["pf", "opf"])
@pytest.mark.parametrize("interrupt_after_skip", [False, True], ids=["retry", "resume"])
def test_timeout_skips_only_offender_and_preserves_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    interrupt_after_skip: bool,
) -> None:
    """Kill a hung real-solver worker and retain exactly the other load scenarios."""
    generate = importlib.import_module("gridfm_datakit.generate")
    config = build_config(
        {
            "load": {"scenarios": 4},
            "settings": {
                "mode": mode,
                "num_processes": 2,
                "large_chunk_size": 4,
                "is_multiprocessing": True,
                "scenario_timeout_sec": 2.0,
                "enable_checkpointing": True,
            },
        },
        str(tmp_path / "baseline"),
    )
    expected = generate.generate_power_flow_data_distributed(config)
    assert (
        json.loads(Path(expected["checkpoint"]).read_text())["skipped_scenarios"] == []
    )
    target = deepcopy(config)
    target["settings"]["data_dir"] = str(tmp_path / "stalled")
    monkeypatch.setattr(generate, "_initialize_scenario_worker", _initialize_with_stall)
    record_skip = GenerationCheckpoint.record_skip

    def interrupt(self: GenerationCheckpoint, index: int) -> None:
        """Simulate interruption immediately after the skip becomes durable."""
        record_skip(self, index)
        raise RuntimeError("interrupted after recording timed-out scenario")

    if interrupt_after_skip:
        monkeypatch.setattr(GenerationCheckpoint, "record_skip", interrupt)
        with pytest.raises(RuntimeError, match="interrupted after recording"):
            generate.generate_power_flow_data_distributed(target)
        raw = tmp_path / "stalled" / config["network"]["name"] / "raw"
        state = json.loads((raw / "checkpoint.json").read_text())
        assert state["skipped_scenarios"] == [2]
        assert state["next_chunk"] == 0
        monkeypatch.setattr(GenerationCheckpoint, "record_skip", record_skip)
        target["settings"]["overwrite"] = False
    actual = generate.generate_power_flow_data_distributed(target)
    state = json.loads(Path(actual["checkpoint"]).read_text())
    assert state["skipped_scenarios"] == [2]
    assert state["next_chunk"] == 1

    for dataset, keys in {
        "bus_data": ["scenario", "bus"],
        "gen_data": ["scenario", "idx"],
        "branch_data": ["scenario", "idx"],
        "y_bus_data": ["scenario", "index1", "index2"],
    }.items():
        baseline = pd.read_parquet(expected[dataset])
        baseline = baseline[baseline["load_scenario_idx"] != 2].copy()
        baseline["scenario"] = pd.factorize(baseline["scenario"], sort=True)[0]
        # Partition ids are derived from the rewritten output scenario numbers.
        baseline["scenario_partition"] = baseline["scenario"] // 200
        observed = pd.read_parquet(actual[dataset])
        baseline["scenario_partition"] = baseline["scenario_partition"].astype(
            observed["scenario_partition"].dtype,
        )
        assert 2 not in observed["load_scenario_idx"].unique()
        assert_frame_equal(
            observed.sort_values(keys).reset_index(drop=True),
            baseline.sort_values(keys).reset_index(drop=True),
            check_exact=True,
            check_categorical=False,
        )
    raw = Path(actual["checkpoint"]).parent
    assert validate_data_directory(str(raw), sn_mva=100.0, n_partitions=0, mode=mode)
    if not interrupt_after_skip:
        records = [
            json.loads(line)
            for line in Path(actual["timeouts_log"]).read_text().splitlines()
        ]
        assert [record["load_scenario_idx"] for record in records] == [2]
