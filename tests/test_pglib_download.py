"""Tests for the PGLib grid download in gridfm_datakit.network."""

from pathlib import Path
from types import SimpleNamespace

from gridfm_datakit import network


class _Response:
    content = b"function mpc = pglib_opf_case14_ieee\nmpc.baseMVA = 100;\n"

    def raise_for_status(self):
        return None


def test_pglib_session_retries_transient_failures():
    session = network._pglib_session()
    retry = session.get_adapter("https://raw.githubusercontent.com").max_retries

    assert retry.total == 4
    assert set(retry.status_forcelist) >= {429, 500, 502, 503, 504}


def test_pglib_download_passes_a_timeout(tmp_path, monkeypatch):
    calls = {}

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def get(self, url, timeout=None):
            calls["url"] = url
            calls["timeout"] = timeout
            return _Response()

    monkeypatch.setattr(network, "_pglib_session", _Session)
    monkeypatch.setattr(network, "correct_network", lambda path: path)
    monkeypatch.setattr(
        network,
        "resources",
        SimpleNamespace(files=lambda pkg: tmp_path),
    )

    file_path = network.get_pglib_file_path("case14_ieee")

    assert calls["timeout"] == (5, 60)
    assert calls["url"].endswith("pglib_opf_case14_ieee.m")
    assert Path(file_path).read_bytes() == _Response.content
