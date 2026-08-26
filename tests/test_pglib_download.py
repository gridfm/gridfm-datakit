"""Tests for the PGLib grid cache and download in gridfm_datakit.network."""

import os
from pathlib import Path

from gridfm_datakit import network


class _Response:
    content = b"function mpc = pglib_opf_case14_ieee\nmpc.baseMVA = 100;\n"

    def raise_for_status(self):
        return None


class _Session:
    calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get(self, url, timeout=None):
        type(self).calls.append({"url": url, "timeout": timeout})
        return _Response()


def _use_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(network.CACHE_DIR_ENV_VAR, str(tmp_path))
    _Session.calls = []
    monkeypatch.setattr(network, "_pglib_session", _Session)


def test_pglib_session_retries_transient_failures():
    session = network._pglib_session()
    retry = session.get_adapter("https://raw.githubusercontent.com").max_retries

    assert retry.total == 4
    assert set(retry.status_forcelist) >= {429, 500, 502, 503, 504}


def test_cache_dir_honours_the_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv(network.CACHE_DIR_ENV_VAR, str(tmp_path))

    cache_dir = Path(network.grids_cache_dir())

    assert cache_dir == tmp_path / "grids"
    assert cache_dir.is_dir()


def test_download_writes_into_the_cache_not_the_package(tmp_path, monkeypatch):
    _use_cache(tmp_path, monkeypatch)

    file_path = Path(network.get_pglib_source_path("case14_ieee"))

    assert file_path == tmp_path / "grids" / "pglib_opf_case14_ieee.m"
    assert file_path.read_bytes() == _Response.content
    assert _Session.calls[0]["timeout"] == (5, 60)
    assert _Session.calls[0]["url"].endswith("pglib_opf_case14_ieee.m")


def test_download_leaves_no_partial_file(tmp_path, monkeypatch):
    _use_cache(tmp_path, monkeypatch)

    network.get_pglib_source_path("case14_ieee")

    leftovers = [n for n in os.listdir(tmp_path / "grids") if n.endswith(".part")]
    assert leftovers == []


def test_cached_file_is_not_downloaded_again(tmp_path, monkeypatch):
    _use_cache(tmp_path, monkeypatch)

    network.get_pglib_source_path("case14_ieee")
    network.get_pglib_source_path("case14_ieee")

    assert len(_Session.calls) == 1
