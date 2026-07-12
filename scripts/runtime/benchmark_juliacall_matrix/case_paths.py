"""Resolve PGLib MATPOWER case files for juliacall benchmarks."""

from __future__ import annotations

from pathlib import Path

from gridfm_datakit.network import correct_network


def pglib_case_file(grids_dir: Path, network: str) -> Path:
    """Return the corrected PGLib case path, creating it on first load if needed."""
    corrected = grids_dir / f"pglib_opf_{network}_corrected.m"
    if corrected.exists():
        return corrected

    uncorrected = grids_dir / f"pglib_opf_{network}.m"
    if not uncorrected.exists():
        raise FileNotFoundError(
            f"PGLib case file not found: {uncorrected} (or {corrected})"
        )

    return Path(correct_network(str(uncorrected)))
