"""
Tests for the Dynawo launcher check in the availability guard.
"""

from gridfm_datakit.dynamic.dynawo import api


def _install(tmp_path, monkeypatch, launcher):
    home = tmp_path / "dynawo"
    home.mkdir()
    if launcher:
        (home / launcher).parent.mkdir(parents=True, exist_ok=True)
        (home / launcher).write_text("")
    config_dir = tmp_path / "itools"
    config_dir.mkdir()
    (config_dir / "config.yml").write_text(f"dynawo:\n  homeDir: {home}\n")
    monkeypatch.setattr(api, "_DEFAULT_CONFIG_DIR", config_dir)


def test_windows_launcher_is_dynawo_cmd():
    assert api._dynawo_launchers("nt") == ("dynawo.cmd",)


def test_posix_launchers():
    assert api._dynawo_launchers("posix") == ("dynawo.sh", "bin/dynawo")


def test_windows_install_is_accepted(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, "dynawo.cmd")
    windows = api._dynawo_launchers("nt")
    monkeypatch.setattr(api, "_dynawo_launchers", lambda: windows)
    assert api._dynawo_unavailable_reason() is None


def test_posix_install_is_accepted(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, "dynawo.sh")
    posix = api._dynawo_launchers("posix")
    monkeypatch.setattr(api, "_dynawo_launchers", lambda: posix)
    assert api._dynawo_unavailable_reason() is None


def test_install_without_launcher_is_rejected(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, None)
    reason = api._dynawo_unavailable_reason()
    assert reason is not None
    assert "launcher" in reason


def test_config_dir_environment_variable_is_ignored(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, "dynawo.sh")
    monkeypatch.setattr(api, "_DEFAULT_CONFIG_DIR", tmp_path / "empty")
    monkeypatch.setenv("POWSYBL_CONFIG_DIR", str(tmp_path / "itools"))
    reason = api._dynawo_unavailable_reason()
    assert reason is not None
    assert "no powsybl configuration file" in reason
