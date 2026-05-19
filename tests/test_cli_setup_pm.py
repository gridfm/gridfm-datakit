"""Regression tests for the setup_pm CLI path."""

import pytest

from gridfm_datakit import cli


def test_pm_setup_worker_uses_subprocess_with_expected_env(monkeypatch, tmp_path):
    """Ensure setup worker invokes Julia through subprocess with stable flags/env."""
    conda_prefix = tmp_path / "conda_prefix"
    julia_project = conda_prefix / "julia_env"
    julia_project.mkdir(parents=True)

    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    monkeypatch.setenv("JULIA_EXE", "C:/fake/julia/bin/julia.exe")

    captured = {}

    def fake_run(cmd, env, check):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["check"] = check

    monkeypatch.setattr("subprocess.run", fake_run)

    cli._pm_setup_worker()

    cmd = captured["cmd"]
    env = captured["env"]

    assert cmd[0] == "C:/fake/julia/bin/julia.exe"
    assert f"--project={julia_project}" in cmd
    assert "--startup-file=no" in cmd
    assert "-e" in cmd
    assert "Pkg.add(pkg)" in cmd[-1]
    assert captured["check"] is True

    assert env["JULIA_PKG_IGNORE_UNKNOWN_REGISTRIES"] == "true"
    assert env["JULIA_SSL_NO_VERIFY_HOSTS"] == "**"
    assert env["JULIA_NO_VERIFY_HOSTS"] == "**"


def test_main_setup_pm_command_path(monkeypatch, capsys):
    """Ensure CLI setup_pm command waits for worker and exits successfully."""

    calls = {"pm_setup": 0, "get": 0, "close": 0, "join": 0}

    class FakeResult:
        def get(self, timeout=None):
            calls["get"] += 1
            assert timeout is None
            return None

    class FakePool:
        def close(self):
            calls["close"] += 1

        def join(self):
            calls["join"] += 1

    def fake_pm_setup():
        calls["pm_setup"] += 1
        return FakePool(), FakeResult()

    monkeypatch.setattr(cli, "pm_setup", fake_pm_setup)
    monkeypatch.setattr("sys.argv", ["gridfm_datakit", "setup_pm"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0
    assert calls == {"pm_setup": 1, "get": 1, "close": 1, "join": 1}

    output = capsys.readouterr().out
    assert "Setting up Julia packages for PowerModels.jl" in output
    assert "Julia packages installed successfully" in output
