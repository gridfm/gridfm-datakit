"""Checkpoint recovery tests using actual parquet appends and simulated interruptions."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import yaml

from gridfm_datakit.checkpoint import GenerationCheckpoint
from gridfm_datakit.generate import _setup_environment
from gridfm_datakit.process import process_network
from gridfm_datakit.utils.utils import write_parquet


_DATASETS = ("bus_data", "branch_data", "gen_data", "y_bus_data", "runtime_data")


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    """Return an auto-seeded checkpoint run with three uneven large chunks."""
    config = yaml.safe_load(Path("scripts/config/default.yaml").read_text(encoding="utf-8"))
    config["settings"].update(data_dir=str(tmp_path / "data"), seed=None, num_processes=1, large_chunk_size=2, enable_checkpointing=True)
    config["load"]["scenarios"] = 5
    return config


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Use the real worker seed policy with inexpensive scenario processing."""
    generate = importlib.import_module("gridfm_datakit.generate")
    monkeypatch.setattr(generate, "_prepare_network_and_scenarios", lambda *_: (object(), np.zeros((1, 5, 2)), {}))
    for name in ("initialize_topology_generator", "initialize_generation_generator", "initialize_admittance_generator"):
        monkeypatch.setattr(generate, name, lambda *_: None)
    monkeypatch.setattr(generate, "write_ram_usage_distributed", lambda *_: None)
    monkeypatch.setattr(process_network, "_worker_context", None)
    monkeypatch.setattr(process_network, "_worker_jl", None)
    monkeypatch.setattr(process_network, "init_julia", Mock(return_value=object()))
    monkeypatch.setattr(type(generate._DEFAULT_SEED_POLICY), "random_base_seed", lambda _: 17)

    def process(*args: Any, **kwargs: Any) -> list:
        """Record a seeded draw and its original load-scenario offset."""
        args[6].append((args[2], float(np.random.random())))
        return args[6]

    monkeypatch.setattr(process_network, "process_scenario_pf_mode", process)
    return generate


def _save_records(data: list, paths: dict[str, str], root: str, fail_during_write: bool = False) -> None:
    """Append actual parquet files and optionally interrupt after one dataset.

    Args:
        data: Seeded scenario records from a simulated solver.
        paths: Generation artifact paths.
        root: Output directory containing sample-count metadata.
        fail_during_write: Whether to fail after writing only the bus dataset.
    """
    count_file = Path(root) / "n_scenarios.txt"
    count = int(count_file.read_text()) if count_file.exists() else 0
    frame = pd.DataFrame(data, columns=["load_scenario_idx", "draw"])
    frame["scenario"] = np.arange(count, count + len(data))
    frame["scenario_partition"] = 0
    for dataset in _DATASETS:
        write_parquet(frame, paths[dataset], partition_cols=["scenario_partition"])
        if fail_during_write:
            raise RuntimeError("interrupted parquet write")
    count_file.write_text(str(count + len(data)), encoding="utf-8")


@pytest.mark.parametrize("interruption", ["during-first-save", "before-save", "during-save", "before-checkpoint", "after-checkpoint"])
def test_resume_matches_uninterrupted_run(
    config: dict[str, Any], pipeline: Any, monkeypatch: pytest.MonkeyPatch, interruption: str,
) -> None:
    """Recover every save boundary without duplicates, omissions, or seed drift."""
    def original_save(net: Any, data: list, paths: dict, root: str, args: Any) -> None:
        """Save every simulated dataset without an interruption."""
        _save_records(data, paths, root)
    monkeypatch.setattr(pipeline, "_save_generated_data", original_save)
    baseline = deepcopy(config)
    baseline["settings"]["data_dir"] += "_baseline"
    expected = pipeline.generate_power_flow_data_distributed(baseline)
    writes = 0

    def interrupted_save(net: Any, data: list, paths: dict, root: str, args: Any) -> None:
        """Interrupt the second large chunk at the selected write boundary."""
        nonlocal writes
        writes += 1
        if writes == 2 and interruption == "before-save":
            raise RuntimeError("interrupted before save")
        partial = (writes == 2 and interruption == "during-save") or (writes == 1 and interruption == "during-first-save")
        _save_records(data, paths, root, partial)

    original_commit = GenerationCheckpoint.commit

    def interrupted_commit(self: GenerationCheckpoint, next_chunk: int) -> None:
        """Simulate interruption immediately before or after publishing state."""
        if next_chunk == 2 and interruption == "before-checkpoint":
            raise RuntimeError("interrupted before checkpoint")
        original_commit(self, next_chunk)
        if next_chunk == 2 and interruption == "after-checkpoint":
            raise RuntimeError("interrupted after checkpoint")

    monkeypatch.setattr(pipeline, "_save_generated_data", interrupted_save)
    monkeypatch.setattr(GenerationCheckpoint, "commit", interrupted_commit)
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.generate_power_flow_data_distributed(config)
    raw = Path(config["settings"]["data_dir"]) / config["network"]["name"] / "raw"
    state = json.loads((raw / "checkpoint.json").read_text())
    assert state["seed"] == 17
    expected_chunk = 0 if interruption == "during-first-save" else 2 if interruption == "after-checkpoint" else 1
    assert state["next_chunk"] == expected_chunk
    if interruption != "after-checkpoint":
        (raw / "n_scenarios.txt").write_text("999", encoding="utf-8")
    monkeypatch.setattr(GenerationCheckpoint, "commit", original_commit)
    monkeypatch.setattr(pipeline, "_save_generated_data", original_save)
    monkeypatch.setattr(type(pipeline._DEFAULT_SEED_POLICY), "random_base_seed", Mock(side_effect=AssertionError("must reuse saved seed")))
    config["settings"]["overwrite"] = False
    resumed = pipeline.generate_power_flow_data_distributed(config)

    for dataset in _DATASETS:
        actual_frame = pd.read_parquet(resumed[dataset]).sort_values("scenario").reset_index(drop=True)
        expected_frame = pd.read_parquet(expected[dataset]).sort_values("scenario").reset_index(drop=True)
        assert_frame_equal(actual_frame, expected_frame, check_exact=True)
        assert actual_frame["scenario"].tolist() == list(range(5))
    assert (raw / "n_scenarios.txt").read_text() == "5"
    assert json.loads((raw / "checkpoint.json").read_text())["next_chunk"] == 3

    prepare = Mock(side_effect=AssertionError("completed run must be a no-op"))
    monkeypatch.setattr(pipeline, "_prepare_network_and_scenarios", prepare)
    pipeline.generate_power_flow_data_distributed(config)
    prepare.assert_not_called()


@pytest.mark.parametrize("setting,value", [("num_processes", 2), ("large_chunk_size", 3), ("seed", 99), ("mode", "opf"), ("enable_checkpointing", False)])
def test_incompatible_resume_does_not_modify_output(config: dict, setting: str, value: Any) -> None:
    """Reject incompatible resumes before rewriting run configuration or logs."""
    args, root, paths, seed = _setup_environment(config, allow_checkpointing=True)
    GenerationCheckpoint.create(Path(root), args.to_dict(), seed)
    original = Path(paths["config"]).read_bytes()
    config["settings"].update(overwrite=False)
    config["settings"][setting] = value
    with pytest.raises(ValueError, match="checkpoint"):
        _setup_environment(config, allow_checkpointing=True)
    assert Path(paths["config"]).read_bytes() == original


def test_resume_refuses_output_without_checkpoint(config: dict) -> None:
    """Do not adopt ordinary append output into a resumable run."""
    raw = Path(config["settings"]["data_dir"]) / config["network"]["name"] / "raw"
    raw.mkdir(parents=True)
    (raw / "n_scenarios.txt").write_text("5", encoding="utf-8")
    config["settings"]["overwrite"] = False
    with pytest.raises(ValueError, match="without checkpoint.json"):
        _setup_environment(config, allow_checkpointing=True)


def test_checkpoint_rejected_by_sequential_api_before_overwrite(config: dict) -> None:
    """Reject unsupported pipelines before touching an existing output directory."""
    with pytest.raises(ValueError, match="only by static"):
        _setup_environment(config)


def test_dynamic_run_cannot_reuse_static_checkpoint_output(config: dict) -> None:
    """Protect checkpointed static input artifacts from a different pipeline."""
    args, root, paths, seed = _setup_environment(config, allow_checkpointing=True)
    GenerationCheckpoint.create(Path(root), args.to_dict(), seed)
    original = Path(paths["config"]).read_bytes()
    config["settings"].update(overwrite=False, enable_checkpointing=False)
    config["dynamic"] = {"solver": "dynawo"}
    with pytest.raises(ValueError, match="Existing output has a checkpoint"):
        _setup_environment(config)
    assert Path(paths["config"]).read_bytes() == original


def test_failed_checkpoint_publication_preserves_previous_state(
    config: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed atomic replace retains the last checkpoint and cleans its temp file."""
    args, root, _, seed = _setup_environment(config, allow_checkpointing=True)
    checkpoint = GenerationCheckpoint.create(Path(root), args.to_dict(), seed)
    original = (Path(root) / "checkpoint.json").read_bytes()
    module = importlib.import_module("gridfm_datakit.checkpoint")
    monkeypatch.setattr(module.os, "replace", Mock(side_effect=OSError("cannot publish checkpoint")))
    with pytest.raises(OSError, match="cannot publish"):
        checkpoint.commit(1)
    assert checkpoint.state.next_chunk == 0
    assert (Path(root) / "checkpoint.json").read_bytes() == original
    assert not list(Path(root).glob(".checkpoint-*"))


@pytest.mark.parametrize("corruption", ["version", "path", "missing-file", "changed-file"])
def test_corrupt_checkpoint_preserves_files(config: dict, corruption: str) -> None:
    """Reject invalid state or committed files without deleting any output."""
    args, root, _, seed = _setup_environment(config, allow_checkpointing=True)
    checkpoint = GenerationCheckpoint.create(Path(root), args.to_dict(), seed)
    state = checkpoint.state.model_dump()
    if corruption == "version":
        state["version"] = 2
    else:
        state.update(next_chunk=1, saved_scenarios=1)
        name = "../outside.parquet" if corruption == "path" else "bus_data.parquet/scenario_partition=0/committed.parquet"
        state["files"] = {name: 4}
        if corruption == "changed-file":
            file = Path(root) / name
            file.parent.mkdir(parents=True)
            file.write_bytes(b"changed")
    (Path(root) / "checkpoint.json").write_text(json.dumps(state), encoding="utf-8")
    config["settings"]["overwrite"] = False
    with pytest.raises(ValueError):
        _setup_environment(config, allow_checkpointing=True)
    assert (Path(root) / "checkpoint.json").exists()
