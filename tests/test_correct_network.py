"""
Test cases for passing network file paths to Julia in correct_network.
"""

from pathlib import Path

import pytest

from gridfm_datakit.network import (
    _julia_string,
    correct_network,
    get_pglib_source_path,
)


@pytest.mark.parametrize(
    "value",
    [
        "/plain/case14.m",
        "C:\\Users\\runner\\AppData\\Local\\case14.m",
        'dir "quoted"/case14.m',
        "dir/$HOME/case14.m",
    ],
)
def test_julia_string_round_trips(value):
    from juliacall import Main as jl

    assert jl.seval(_julia_string(value)) == value


def test_correct_network_handles_special_characters_in_path(tmp_path):
    directory = tmp_path / 'grid $HOME "dir"'
    directory.mkdir()
    source = directory / "case14_ieee.m"
    with open(get_pglib_source_path("case14_ieee"), encoding="utf-8") as f:
        source.write_text(f.read(), encoding="utf-8")

    corrected = correct_network(str(source))

    assert corrected == str(directory / "case14_ieee_corrected.m")
    assert Path(corrected).stat().st_size > 0
