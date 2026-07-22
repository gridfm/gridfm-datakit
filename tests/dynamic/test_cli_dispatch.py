"""The `generate` CLI command routes on the presence of a dynamic: block.

Needs neither pypowsybl nor Dynawo: the routing decision is made from the config
file alone, and the pipelines themselves are stubbed out.
"""

from __future__ import annotations

import sys

import pytest
import yaml

from gridfm_datakit import cli


STATIC_CONFIG = {
    "network": {"name": "IEEE14", "reader": "powsybl"},
    "settings": {"data_dir": "out"},
}
DYNAMIC_CONFIG = {
    **STATIC_CONFIG,
    "dynamic": {"dynamic_solver": "dynawo"},
}


def _write(tmp_path, config, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return str(path)


class TestConfigRouting:
    def test_dynamic_block_selects_the_dynamic_pipeline(self, tmp_path):
        assert cli._config_has_dynamic_block(_write(tmp_path, DYNAMIC_CONFIG)) is True

    def test_absent_dynamic_block_selects_the_static_pipeline(self, tmp_path):
        assert cli._config_has_dynamic_block(_write(tmp_path, STATIC_CONFIG)) is False

    def test_empty_dynamic_block_is_not_a_dynamic_run(self, tmp_path):
        # "dynamic:" with nothing under it parses to None; it names no solver, so
        # routing to the dynamic pipeline would only produce a confusing error.
        config = {**STATIC_CONFIG, "dynamic": None}
        assert cli._config_has_dynamic_block(_write(tmp_path, config)) is False

    @pytest.mark.parametrize("content", ["", "just a string", "[1, 2, 3]"])
    def test_unusable_config_falls_through_to_the_static_pipeline(
        self,
        tmp_path,
        content,
    ):
        # The pipeline reports the problem with its own message; routing must not
        # raise first with a less useful one.
        path = tmp_path / "config.yaml"
        path.write_text(content)
        assert cli._config_has_dynamic_block(str(path)) is False

    def test_missing_file_does_not_raise_during_routing(self, tmp_path):
        assert cli._config_has_dynamic_block(str(tmp_path / "nope.yaml")) is False


class TestGenerateCommand:
    """The dispatch itself, with both pipelines stubbed."""

    @pytest.fixture
    def stub_pipelines(self, monkeypatch):
        calls = {}

        def _static(config):
            calls["static"] = config
            return {"bus_data": "static.parquet"}

        def _dynamic(config):
            calls["dynamic"] = config
            return {"bus_data": "dynamic.parquet"}

        monkeypatch.setattr(cli, "generate_power_flow_data_distributed", _static)
        module = pytest.importorskip(
            "gridfm_datakit.dynamic.generate_dynamic",
            reason="dynamic module unavailable",
        )
        monkeypatch.setattr(module, "generate_dynamic_data", _dynamic)
        return calls

    def test_dynamic_config_calls_the_dynamic_pipeline(
        self,
        tmp_path,
        monkeypatch,
        stub_pipelines,
    ):
        path = _write(tmp_path, DYNAMIC_CONFIG)
        monkeypatch.setattr(sys, "argv", ["gridfm-datakit", "generate", path])
        cli.main()
        assert stub_pipelines == {"dynamic": path}

    def test_static_config_calls_the_static_pipeline(
        self,
        tmp_path,
        monkeypatch,
        stub_pipelines,
    ):
        path = _write(tmp_path, STATIC_CONFIG)
        monkeypatch.setattr(sys, "argv", ["gridfm-datakit", "generate", path])
        cli.main()
        assert stub_pipelines == {"static": path}
