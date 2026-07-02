#!/usr/bin/env python3
"""Plot per-sample wall time vs worker count for all grids and solvers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
MODES = ("pf", "dcpf", "opf", "dcopf")
MODE_TITLES = {
    "pf": "PF",
    "dcpf": "DC-PF",
    "opf": "OPF",
    "dcopf": "DC-OPF",
}

GRID_ORDER = [
    "case14_ieee",
    "case30_ieee",
    "case57_ieee",
    "case118_ieee",
    "case500_goc",
    "case2000_goc",
    "case10000_goc",
]

GRID_COLORS = {
    14: "#1f77b4",
    30: "#17becf",
    57: "#ff7f0e",
    118: "#bcbd22",
    500: "#d62728",
    2000: "#2ca02c",
    10000: "#8c564b",
}

AXIS_LABEL_FONT_SIZE = 17
TICK_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 16
LEGEND_TITLE_FONT_SIZE = 18
PANEL_TITLE_FONT_SIZE = 20
PANEL_WIDTH = 5
PANEL_HEIGHT = 7.0
WORKER_TICKS = list(range(24, 217, 24))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Defaults to <output-root>/juliacall_scaling_comparison.pdf",
    )
    parser.add_argument(
        "--linear-x",
        action="store_true",
        help="Use linear scale on x-axis (default: log).",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use linear scale on y-axis (default: log).",
    )
    return parser.parse_args()


def parse_name(path: Path) -> tuple[str, str] | None:
    match = re.match(r"benchmark_(.+)_(pf|dcpf|opf|dcopf)\.csv$", path.name)
    if match is None:
        return None
    return match.group(1), match.group(2)


def bus_count(grid: str) -> int:
    match = re.search(r"case(\d+)_", grid)
    if match is None:
        raise ValueError(f"Cannot parse bus count from grid name: {grid}")
    return int(match.group(1))


def collect_series(output_root: Path) -> pd.DataFrame:
    best_files: dict[tuple[str, str], tuple[int, Path]] = {}

    for csv_path in sorted(output_root.rglob("benchmark_*.csv")):
        if "_pf_fast" in csv_path.name:
            continue

        parsed = parse_name(csv_path)
        if parsed is None:
            continue

        grid, mode = parsed
        if mode not in MODES:
            continue

        key = (grid, mode)
        n_pfs = int(pd.read_csv(csv_path, usecols=["n_pfs"]).iloc[0, 0])
        if key not in best_files or n_pfs > best_files[key][0]:
            best_files[key] = (n_pfs, csv_path)

    records: list[dict[str, float | int | str]] = []
    for (grid, mode), (_, csv_path) in sorted(best_files.items()):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            records.append(
                {
                    "grid": grid,
                    "mode": mode,
                    "p": int(row["p"]),
                    "n_pfs": int(row["n_pfs"]),
                    "per_sample_ms": float(row["pf_elapsed_s"]) / float(row["n_pfs"]) * 1000.0,
                    "buses": bus_count(grid),
                }
            )

    if not records:
        raise RuntimeError(f"No benchmark CSVs found under {output_root}")

    return pd.DataFrame(records)


def ordered_grids(data: pd.DataFrame) -> list[str]:
    grids = [grid for grid in GRID_ORDER if grid in data["grid"].unique()]
    grids.extend(sorted(set(data["grid"]) - set(grids)))
    return grids


def plot_mode_panel(
    axis: plt.Axes,
    data: pd.DataFrame,
    mode: str,
    *,
    log_x: bool,
    log_y: bool,
    show_ylabel: bool,
    add_legend_labels: bool,
) -> None:
    grids = ordered_grids(data)
    mode_data = data[data["mode"] == mode]

    for grid in grids:
        series = mode_data[mode_data["grid"] == grid].sort_values("p")
        if series.empty:
            continue

        buses = int(series["buses"].iloc[0])
        label = str(buses) if add_legend_labels else "_nolegend_"

        axis.plot(
            series["p"],
            series["per_sample_ms"],
            marker="o",
            linewidth=2.2,
            markersize=7,
            linestyle="-",
            color=GRID_COLORS.get(buses),
            label=label,
        )

    if log_x:
        axis.set_xscale("log")
    if log_y:
        axis.set_yscale("log")

    axis.set_xlim(WORKER_TICKS[0], WORKER_TICKS[-1])
    axis.xaxis.set_major_locator(FixedLocator(WORKER_TICKS))
    axis.xaxis.set_major_formatter(FixedFormatter([str(tick) for tick in WORKER_TICKS]))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xlabel("Number of workers [-]", fontsize=AXIS_LABEL_FONT_SIZE)
    if show_ylabel:
        axis.set_ylabel("Time per solved instance [ms]", fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_title(MODE_TITLES[mode], fontsize=PANEL_TITLE_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis.tick_params(axis="x", rotation=90)
    axis.grid(True, which="both", alpha=0.2)


def plot_by_mode(
    data: pd.DataFrame,
    plot_path: Path,
    *,
    log_x: bool,
    log_y: bool,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(MODES),
        figsize=(PANEL_WIDTH * len(MODES), PANEL_HEIGHT),
        sharex=True,
        sharey=True,
    )
    if len(MODES) == 1:
        axes = [axes]

    for index, (axis, mode) in enumerate(zip(axes, MODES, strict=True)):
        plot_mode_panel(
            axis,
            data,
            mode,
            log_x=log_x,
            log_y=log_y,
            show_ylabel=index == 0,
            add_legend_labels=index == 0,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Bus count",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(labels),
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_path = args.plot_path or (args.output_root / "juliacall_scaling_comparison.pdf")

    data = collect_series(args.output_root)
    plot_by_mode(
        data,
        plot_path,
        log_x=not args.linear_x,
        log_y=not args.linear_y,
    )
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
