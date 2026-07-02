#!/usr/bin/env python3
"""Stacked barplot: solver time vs overhead in per-job runtime."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
MODES = ("pf", "dcpf", "opf", "dcopf")

GRID_ORDER = [
    "case14_ieee",
    "case30_ieee",
    "case57_ieee",
    "case118_ieee",
    "case500_goc",
    "case2000_goc",
    "case10000_goc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Defaults to <output-root>/solve_time_contribution.png",
    )
    parser.add_argument(
        "--target-p",
        type=int,
        default=72,
        help="Process count row to use (default: 72). Ignored if --pick-best-p.",
    )
    parser.add_argument(
        "--pick-best-p",
        action="store_true",
        help="Use the row with lowest mean_pf_runtime_s per CSV.",
    )
    return parser.parse_args()


def parse_name(path: Path) -> tuple[str, str] | None:
    match = re.match(r"benchmark_(.+)_(pf|dcpf|opf|dcopf)\.csv$", path.name)
    if match is None:
        return None
    return match.group(1), match.group(2)


def load_row(df: pd.DataFrame, target_p: int | None, pick_best_p: bool) -> pd.Series:
    if pick_best_p:
        return df.loc[df["mean_pf_runtime_s"].idxmin()]

    if target_p is not None:
        row = df[df["p"] == target_p]
        if not row.empty:
            return row.iloc[0]

    return df.iloc[len(df) // 2]


def collect_records(
    output_root: Path,
    target_p: int | None,
    pick_best_p: bool,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []

    for csv_path in sorted(output_root.rglob("benchmark_*.csv")):
        if "_pf_fast" in csv_path.name:
            continue

        parsed = parse_name(csv_path)
        if parsed is None:
            continue

        grid, mode = parsed
        if mode not in MODES:
            continue

        df = pd.read_csv(csv_path)
        row = load_row(df, target_p=target_p, pick_best_p=pick_best_p)
        runtime = float(row["mean_pf_runtime_s"])
        solve = float(row["mean_pf_solve_time_s"])
        records.append(
            {
                "grid": grid,
                "mode": mode,
                "runtime": runtime,
                "solve": solve,
                "overhead": max(runtime - solve, 0.0),
                "fraction": solve / runtime if runtime > 0 else 0.0,
                "p": int(row["p"]),
            }
        )

    if not records:
        raise RuntimeError(f"No benchmark CSVs found under {output_root}")

    return pd.DataFrame(records)


def plot_contribution(data: pd.DataFrame, plot_path: Path) -> None:
    grids = [grid for grid in GRID_ORDER if grid in data["grid"].unique()]
    grids.extend(sorted(set(data["grid"]) - set(grids)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes_list = axes.ravel()

    for axis, mode in zip(axes_list, MODES, strict=True):
        subset = data[data["mode"] == mode].set_index("grid").reindex(grids)
        subset = subset.dropna(subset=["runtime"])
        if subset.empty:
            axis.set_title(f"{mode} (no data)")
            continue

        x_positions = range(len(subset))
        labels = [
            grid.replace("case", "").replace("_ieee", "").replace("_goc", "")
            for grid in subset.index
        ]

        axis.bar(x_positions, subset["solve"], label="solver (solve_time)", color="#2ca02c")
        axis.bar(
            x_positions,
            subset["overhead"],
            bottom=subset["solve"],
            label="overhead",
            color="#ff7f0e",
        )

        for index, (_, row) in enumerate(subset.iterrows()):
            axis.text(
                index,
                row["runtime"],
                f"{100 * row['fraction']:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        axis.set_xticks(list(x_positions))
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_ylabel("mean per-job runtime (s)")
        if subset["p"].nunique() == 1:
            p_note = f"p={subset['p'].iloc[0]}"
        else:
            p_note = "mixed p"
        axis.set_title(f"{mode} ({p_note})")
        axis.grid(axis="y", linestyle="--", alpha=0.3)

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Contribution of solver time to per-job runtime", y=1.06, fontsize=13)
    fig.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_path = args.plot_path or (args.output_root / "solve_time_contribution.png")

    target_p = None if args.pick_best_p else args.target_p
    data = collect_records(
        output_root=args.output_root,
        target_p=target_p,
        pick_best_p=args.pick_best_p,
    )
    plot_contribution(data, plot_path)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
