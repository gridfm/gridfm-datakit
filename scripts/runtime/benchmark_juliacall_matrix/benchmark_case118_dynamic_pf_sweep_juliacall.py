#!/usr/bin/env python3
"""Benchmark dynamic PF/OPF scheduling in Python using juliacall."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from statistics import mean


JULIA_ENV = "/u/apu/gridfm-datakit/venv/julia_env"
DEFAULT_CASE_FILE = (
    "/u/apu/gridfm-datakit/gridfm_datakit/grids/pglib_opf_case14_ieee_corrected.m"
)
DEFAULT_OUTPUT_CSV = (
    "/u/apu/gridfm-datakit/scripts/case118_dynamic_pf_sweep_juliacall_results.csv"
)
DEFAULT_PLOTS_DIR = (
    "/u/apu/gridfm-datakit/scripts/case118_dynamic_pf_sweep_juliacall_plots"
)
DEFAULT_N_PFS = 1_000_000
DEFAULT_P_START = 24
DEFAULT_P_STOP = 216
DEFAULT_P_STEP = 16
DEFAULT_MAX_ITER = 100
DEFAULT_TOL = 1e-6
DEFAULT_PRINT_LEVEL = 0
DEFAULT_INIT_TIMEOUT_S = 900.0
BENCHMARK_MODES = ("pf", "dcpf", "opf", "dcopf")

_WORKER_JL = None
_WORKER_MODE = ""
_WORKER_PF_FAST = True
_WORKER_MAX_ITER = DEFAULT_MAX_ITER
_WORKER_TOL = DEFAULT_TOL
_WORKER_PRINT_LEVEL = DEFAULT_PRINT_LEVEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-file", default=DEFAULT_CASE_FILE)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--mode", choices=BENCHMARK_MODES, default="pf")
    parser.add_argument("--n-pfs", type=int, default=DEFAULT_N_PFS)
    parser.add_argument("--process-start", type=int, default=DEFAULT_P_START)
    parser.add_argument("--process-stop", type=int, default=DEFAULT_P_STOP)
    parser.add_argument("--process-step", type=int, default=DEFAULT_P_STEP)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--print-level", type=int, default=DEFAULT_PRINT_LEVEL)
    parser.add_argument("--init-timeout-s", type=float, default=DEFAULT_INIT_TIMEOUT_S)
    parser.add_argument(
        "--pf-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use compute_ac_pf instead of Ipopt-based solve_ac_pf (pf mode only).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmarking and only generate plots from the CSV.",
    )
    return parser.parse_args()


def _configure_juliacall_env() -> None:
    julia_exe = shutil.which("julia")
    if julia_exe is None:
        raise RuntimeError("Could not find 'julia' on PATH")

    os.environ.setdefault("JULIA_PROJECT", JULIA_ENV)
    os.environ.setdefault("PYTHON_JULIACALL_PROJECT", JULIA_ENV)
    os.environ.setdefault("PYTHON_JULIACALL_EXE", julia_exe)


def _define_julia_helpers(jl) -> None:
    jl.seval(
        f"""
        using Pkg
        Pkg.activate("{JULIA_ENV}"; io=devnull)
        using PowerModels
        using Ipopt
        using Memento
        Memento.config!("not_set")

        if !isdefined(Main, :PY_BENCH_BASE_NETWORK)
            const PY_BENCH_BASE_NETWORK = Ref{{Any}}(nothing)
        end

        if !isdefined(Main, :py_bench_init_case)
            function py_bench_init_case(case_file)
                PY_BENCH_BASE_NETWORK[] = PowerModels.parse_file(case_file)
                return nothing
            end
        end

        if !isdefined(Main, :py_bench_ipopt)
            function py_bench_ipopt(max_iter, tol, print_level)
                return optimizer_with_attributes(
                    Ipopt.Optimizer,
                    "tol" => tol,
                    "print_level" => print_level,
                    "max_iter" => max_iter,
                )
            end
        end

        if !isdefined(Main, :py_bench_run)
            function py_bench_run(mode, pf_fast, max_iter, tol, print_level)
                PY_BENCH_BASE_NETWORK[] === nothing && error("Worker network not initialized")

                network = PY_BENCH_BASE_NETWORK[]
                if mode == "pf"
                    if pf_fast
                        result = compute_ac_pf(network)
                        if result["termination_status"] == false
                            error("PF failed")
                        end
                    else
                        result = solve_ac_pf(network, py_bench_ipopt(max_iter, tol, print_level))
                        if string(result["termination_status"]) != "LOCALLY_SOLVED"
                            error("PF failed with status $(result["termination_status"])")
                        end
                    end
                elseif mode == "dcpf"
                    result = compute_dc_pf(network)
                    if result["termination_status"] == false
                        error("DC PF failed")
                    end
                elseif mode == "opf"
                    result = solve_ac_opf(network, py_bench_ipopt(max_iter, tol, print_level))
                    if string(result["termination_status"]) != "LOCALLY_SOLVED"
                        error("OPF failed with status $(result["termination_status"])")
                    end
                elseif mode == "dcopf"
                    result = solve_dc_opf(network, py_bench_ipopt(max_iter, tol, print_level))
                    if string(result["termination_status"]) != "LOCALLY_SOLVED"
                        error("DC OPF failed with status $(result["termination_status"])")
                    end
                else
                    error("Unknown benchmark mode: $(mode)")
                end

                if !haskey(result, "solve_time")
                    error("Solver result did not contain solve_time")
                end

                return Float64(result["solve_time"])
            end
        end

        if !isdefined(Main, :py_bench_prepare_opf_network)
            function py_bench_prepare_opf_network(case_file, network_file, max_iter, tol, print_level)
                data = PowerModels.parse_file(case_file)
                result = solve_ac_opf(
                    data,
                    py_bench_ipopt(max_iter, tol, print_level),
                )

                status = string(result["termination_status"])
                if status != "LOCALLY_SOLVED"
                    error("OPF failed with status $(status)")
                end

                PowerModels.update_data!(data, result["solution"])
                PowerModels.export_file(network_file, data)

                if !haskey(result, "solve_time")
                    error("OPF result did not contain solve_time")
                end

                return Float64(result["solve_time"])
            end
        end
        """
    )


def prepare_opf_network(
    case_file: str,
    network_file: str,
    max_iter: int,
    tol: float,
    print_level: int,
) -> float:
    _configure_juliacall_env()
    from juliacall import Main as jl

    _define_julia_helpers(jl)
    started_at = time.perf_counter()
    solve_time_s = float(
        jl.py_bench_prepare_opf_network(case_file, network_file, max_iter, tol, print_level)
    )
    elapsed_s = time.perf_counter() - started_at
    print(
        f"OPF solved in {elapsed_s:.3f} s (solver solve_time={solve_time_s:.6f} s); "
        f"updated network written to {network_file}",
        flush=True,
    )
    return elapsed_s


def _worker_init(
    case_file: str,
    mode: str,
    pf_fast: bool,
    max_iter: int,
    tol: float,
    print_level: int,
) -> None:
    global _WORKER_JL
    global _WORKER_MODE
    global _WORKER_PF_FAST
    global _WORKER_MAX_ITER
    global _WORKER_TOL
    global _WORKER_PRINT_LEVEL

    _WORKER_MODE = mode
    _WORKER_PF_FAST = pf_fast
    _WORKER_MAX_ITER = max_iter
    _WORKER_TOL = tol
    _WORKER_PRINT_LEVEL = print_level

    _configure_juliacall_env()
    from juliacall import Main as jl

    _define_julia_helpers(jl)
    jl.py_bench_init_case(case_file)
    jl.py_bench_run(mode, pf_fast, max_iter, tol, print_level)
    _WORKER_JL = jl


def _solve_one_job(_: int) -> tuple[str, float, float]:
    if _WORKER_JL is None:
        raise RuntimeError("Worker Julia runtime not initialized")

    started_at = time.perf_counter()
    solve_time_s = float(
        _WORKER_JL.py_bench_run(
            _WORKER_MODE,
            _WORKER_PF_FAST,
            _WORKER_MAX_ITER,
            _WORKER_TOL,
            _WORKER_PRINT_LEVEL,
        )
    )
    elapsed_s = time.perf_counter() - started_at
    return mp.current_process().name, elapsed_s, solve_time_s


def _worker_ping(_: int) -> str:
    if _WORKER_JL is None:
        raise RuntimeError("Worker Julia runtime not initialized")
    # Give other freshly spawned workers time to come online so the pool
    # distributes warmup tasks across distinct processes.
    time.sleep(0.05)
    return mp.current_process().name


def write_results_csv(path: str | Path, results: list[dict[str, float | int | str]]) -> None:
    with open(path, "w") as f:
        f.write(
            "p,opf_elapsed_s,init_elapsed_s,pf_elapsed_s,n_pfs,total_completed,"
            "min_pf_runtime_s,mean_pf_runtime_s,max_pf_runtime_s,"
            "min_pf_solve_time_s,mean_pf_solve_time_s,max_pf_solve_time_s,"
            "min_worker_completed,mean_worker_completed,max_worker_completed\n"
        )
        for result in results:
            f.write(
                f"{result['p']},{result['opf_elapsed_s']:.6f},{result['init_elapsed_s']:.6f},"
                f"{result['pf_elapsed_s']:.6f},"
                f"{result['n_pfs']},{result['total_completed']},"
                f"{result['min_pf_runtime_s']:.6f},{result['mean_pf_runtime_s']:.6f},"
                f"{result['max_pf_runtime_s']:.6f},"
                f"{result['min_pf_solve_time_s']:.6f},{result['mean_pf_solve_time_s']:.6f},"
                f"{result['max_pf_solve_time_s']:.6f},"
                f"{result['min_worker_completed']},{result['mean_worker_completed']:.6f},"
                f"{result['max_worker_completed']}\n"
            )


def load_results_csv(path: str | Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "p": float(row["p"]),
                    "pf_elapsed_s": float(row["pf_elapsed_s"]),
                    "mean_pf_runtime_s": float(row["mean_pf_runtime_s"]),
                    "mean_pf_solve_time_s": float(row["mean_pf_solve_time_s"]),
                }
            )

    if not rows:
        raise RuntimeError(f"No rows found in results CSV: {path}")

    rows.sort(key=lambda row: row["p"])
    return rows


def plot_results(csv_path: str | Path, plots_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    rows = load_results_csv(csv_path)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    p_values = [int(row["p"]) for row in rows]
    plot_specs = [
        (
            "p_vs_pf_elapsed_s.png",
            "PF elapsed time (s)",
            "p vs pf_elapsed_s",
            [row["pf_elapsed_s"] for row in rows],
        ),
        (
            "p_vs_mean_pf_runtime_s.png",
            "Mean PF runtime (s)",
            "p vs mean_pf_runtime",
            [row["mean_pf_runtime_s"] for row in rows],
        ),
        (
            "p_vs_mean_pf_runtime_s_per_process.png",
            "Mean PF runtime / p (s)",
            "p vs mean_pf_runtime / p",
            [row["mean_pf_runtime_s"] / row["p"] for row in rows],
        ),
        (
            "p_vs_mean_pf_solve_time_s.png",
            "Mean PF solve time (s)",
            "p vs mean_pf_solve_time",
            [row["mean_pf_solve_time_s"] for row in rows],
        ),
        (
            "p_vs_mean_pf_solve_time_s_per_process.png",
            "Mean PF solve time / p (s)",
            "p vs mean_pf_solve_time / p",
            [row["mean_pf_solve_time_s"] / row["p"] for row in rows],
        ),
    ]

    for filename, ylabel, title, y_values in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(p_values, y_values, marker="o", linewidth=2)
        ax.set_xlabel("p")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        output_path = plots_dir / filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {output_path}")


def benchmark_for_p(
    case_file: str,
    mode: str,
    pf_fast: bool,
    opf_elapsed_s: float,
    n_pfs: int,
    p: int,
    max_iter: int,
    tol: float,
    print_level: int,
    init_timeout_s: float,
) -> dict[str, float | int]:
    print()
    pf_fast_label = f" pf_fast={pf_fast}" if mode == "pf" else ""
    print(f"=== Benchmarking mode={mode}{pf_fast_label} p={p} ===", flush=True)

    ctx = mp.get_context("spawn")
    worker_counts: Counter[str] = Counter()
    pf_runtimes: list[float] = []
    pf_solve_times: list[float] = []

    initargs = (case_file, mode, pf_fast, max_iter, tol, print_level)
    with ctx.Pool(processes=p, initializer=_worker_init, initargs=initargs) as pool:
        init_started_at = time.perf_counter()
        seen_workers: set[str] = set()
        next_ping_id = 0
        init_batch_size = max(4 * p, 16)

        while len(seen_workers) < p:
            elapsed = time.perf_counter() - init_started_at
            if elapsed > init_timeout_s:
                raise RuntimeError(
                    f"Timed out waiting for {p} initialized workers; "
                    f"only saw {len(seen_workers)} after {elapsed:.1f}s"
                )

            batch = range(next_ping_id, next_ping_id + init_batch_size)
            next_ping_id += init_batch_size
            for worker_name in pool.imap_unordered(_worker_ping, batch, chunksize=1):
                seen_workers.add(worker_name)
                if len(seen_workers) == p:
                    break
        init_elapsed_s = time.perf_counter() - init_started_at

        pf_started_at = time.perf_counter()
        for worker_name, runtime_s, solve_time_s in pool.imap_unordered(
            _solve_one_job,
            range(n_pfs),
            chunksize=1,
        ):
            worker_counts[worker_name] += 1
            pf_runtimes.append(runtime_s)
            pf_solve_times.append(solve_time_s)

        pf_elapsed_s = time.perf_counter() - pf_started_at

    total_completed = sum(worker_counts.values())
    if total_completed != n_pfs:
        raise RuntimeError(f"Expected {n_pfs} solves, got {total_completed}")

    worker_completed = list(worker_counts.values())
    result = {
        "p": p,
        "opf_elapsed_s": opf_elapsed_s,
        "init_elapsed_s": init_elapsed_s,
        "pf_elapsed_s": pf_elapsed_s,
        "n_pfs": n_pfs,
        "total_completed": total_completed,
        "min_pf_runtime_s": min(pf_runtimes),
        "mean_pf_runtime_s": mean(pf_runtimes),
        "max_pf_runtime_s": max(pf_runtimes),
        "min_pf_solve_time_s": min(pf_solve_times),
        "mean_pf_solve_time_s": mean(pf_solve_times),
        "max_pf_solve_time_s": max(pf_solve_times),
        "min_worker_completed": min(worker_completed),
        "mean_worker_completed": mean(worker_completed),
        "max_worker_completed": max(worker_completed),
    }

    print(
        f"mode={mode}{pf_fast_label} | p={p} | opf_time={opf_elapsed_s:.3f} s | "
        f"init_time={init_elapsed_s:.3f} s | "
        f"solve_wall_time={pf_elapsed_s:.3f} s | total_solves={total_completed}",
        flush=True,
    )
    print(
        "solve runtime min/mean/max = "
        f"{result['min_pf_runtime_s']:.6f} / {result['mean_pf_runtime_s']:.6f} / "
        f"{result['max_pf_runtime_s']:.6f} s",
        flush=True,
    )
    print(
        "solve_time min/mean/max = "
        f"{result['min_pf_solve_time_s']:.6f} / {result['mean_pf_solve_time_s']:.6f} / "
        f"{result['max_pf_solve_time_s']:.6f} s",
        flush=True,
    )
    print(
        "worker completed min/mean/max = "
        f"{result['min_worker_completed']} / {result['mean_worker_completed']:.2f} / "
        f"{result['max_worker_completed']}",
        flush=True,
    )

    return result


def main() -> None:
    args = parse_args()
    if args.plot_only:
        plot_results(args.output_csv, args.plots_dir)
        return

    process_counts = range(args.process_start, args.process_stop + 1, args.process_step)

    print(f"Mode: {args.mode}")
    print(f"Case file: {args.case_file}")
    if args.mode == "pf":
        print(f"PF fast: {args.pf_fast}")
    print(f"Total solves per sweep point: {args.n_pfs}")
    print(f"Process counts: {list(process_counts)}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Plots dir: {args.plots_dir}")

    opf_network_file: str | None = None
    results: list[dict[str, float | int]] = []
    try:
        if args.mode in ("pf", "dcpf"):
            tmp_fd, opf_network_file = tempfile.mkstemp(
                suffix=".json",
                prefix="py_bench_opf_network_",
            )
            os.close(tmp_fd)
            opf_elapsed_s = prepare_opf_network(
                case_file=args.case_file,
                network_file=opf_network_file,
                max_iter=args.max_iter,
                tol=args.tol,
                print_level=args.print_level,
            )
            worker_case_file = opf_network_file
        else:
            opf_elapsed_s = 0.0
            worker_case_file = args.case_file

        for p in process_counts:
            result = benchmark_for_p(
                case_file=worker_case_file,
                mode=args.mode,
                pf_fast=args.pf_fast,
                opf_elapsed_s=opf_elapsed_s,
                n_pfs=args.n_pfs,
                p=p,
                max_iter=args.max_iter,
                tol=args.tol,
                print_level=args.print_level,
                init_timeout_s=args.init_timeout_s,
            )
            results.append(result)
            write_results_csv(args.output_csv, results)

        plot_results(args.output_csv, args.plots_dir)
    finally:
        if opf_network_file is not None and os.path.exists(opf_network_file):
            os.unlink(opf_network_file)

    print()
    print(f"Finished sweep. Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
