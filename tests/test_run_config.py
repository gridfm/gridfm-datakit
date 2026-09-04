"""Tests for run configuration recovery in gridfm_datakit.cli."""

import yaml

from gridfm_datakit.cli import _read_run_config

CONFIG = {"settings": {"mode": "pf"}, "load": {"scenarios": 10}}


def _append_run(path, mode):
    config = {"settings": {"mode": mode}, "load": {"scenarios": 10}}
    with open(path, "a") as f:
        f.write("\nNew generation started at 2026-08-26 10:00:00\n")
        yaml.safe_dump(config, f)


def test_config_yaml_is_preferred(tmp_path):
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.safe_dump(CONFIG, f)
    _append_run(tmp_path / "args.log", "opf")

    source, args = _read_run_config(tmp_path)

    assert source == "config.yaml"
    assert args["settings"]["mode"] == "pf"


def test_args_log_fallback_for_a_single_run(tmp_path):
    _append_run(tmp_path / "args.log", "opf")

    source, args = _read_run_config(tmp_path)

    assert source == "args.log"
    assert args["settings"]["mode"] == "opf"


def test_args_log_fallback_uses_the_last_run(tmp_path):
    _append_run(tmp_path / "args.log", "pf")
    _append_run(tmp_path / "args.log", "opf")

    source, args = _read_run_config(tmp_path)

    assert source == "args.log"
    assert args["settings"]["mode"] == "opf"


def test_missing_configuration_reports_nothing(tmp_path):
    source, args = _read_run_config(tmp_path)

    assert args is None
