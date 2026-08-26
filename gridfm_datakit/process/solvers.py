"""
Power flow and optimal power flow solvers using Julia interface.

This module provides functions for running power flow (PF), optimal power flow (OPF),
DC power flow (DCPF), and DC optimal power flow (DC OPF) calculations using Julia's PowerModels.jl package.
It also includes functionality for comparing results between temporary files
and original case files.
"""

import hashlib
import numpy as np
import tempfile
import os
from typing import Any, Dict
from gridfm_datakit.network import Network
from gridfm_datakit.process.solver_output import solver_capture
from gridfm_datakit.utils.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    RATE_A,
    RATE_B,
    RATE_C,
    SHIFT,
    TAP,
)
from gridfm_datakit.utils.idx_bus import BUS_I, BUS_TYPE, PD, QD, VA, VM
from gridfm_datakit.utils.idx_cost import COST, NCOST
from gridfm_datakit.utils.idx_gen import GEN_STATUS, PG, QG, VG
from typing import Union

# Fingerprint of the base network whose parsed dict currently lives in the
# Julia process (_GFM_BASE). One per process: jl (juliacall Main) is a
# process-wide singleton, so a plain module global mirrors its state.
_ACTIVE_BASE_KEY = None


def _network_fingerprint(net: Network) -> str:
    """Identity of the base case a Network was constructed from.

    Perturbations mutate ``net.buses``/``net.gens``/... but never the original
    ``net.mpc`` arrays, so hashing those identifies the base network even on a
    heavily perturbed copy.
    """
    h = hashlib.sha1(usedforsecurity=False)
    h.update(np.float64(net.baseMVA).tobytes())
    for key in ("bus", "gen", "branch", "gencost"):
        h.update(np.ascontiguousarray(net.mpc[key], dtype=np.float64).tobytes())
    return h.hexdigest()


_LIMIT_COLS = [RATE_A, RATE_B, RATE_C, TAP, SHIFT, ANGMIN, ANGMAX]


def _branch_limit_fingerprint(net: Network) -> bytes:
    """Hash of the branch fields ``_gfm_state`` does NOT push per solve.

    Ratings, tap, shift, and angle limits live on ``net.branches`` just like
    R/X/B/status, but only R/X/B/status get pushed into the reusable Julia
    work state on every call. Unlike ``_network_fingerprint`` this must be
    recomputed every call (not cached on ``net``), so a change to these
    fields — e.g. a limit study lowering ``net.branches[:, RATE_A]`` — is
    detected and forces a reparse instead of silently reusing stale limits.
    """
    return np.ascontiguousarray(
        net.branches[:, _LIMIT_COLS],
        dtype=np.float64,
    ).tobytes()


def _julia_pm_data(net: Network, jl: Any) -> Any:
    """Build a PowerModels data dict for ``net`` in-memory in Julia.

    The MATPOWER case is written to a file and parsed only once per
    (process, base network, branch ratings/tap/shift/angle limits); after
    that each call resets one process-local work dictionary with the mutable
    state (loads, setpoints, statuses, admittances, costs). The returned object
    is valid only until the next call in this worker; solver calls are serial.
    Its contents are field-for-field equivalent to
    ``PowerModels.parse_file(net.to_mpc(...))`` (see
    tests/test_pm_data_path.py), without per-solve copying, serialization, or
    parsing.
    """
    global _ACTIVE_BASE_KEY
    solver_cache = getattr(net, "_solver_cache", None)
    if solver_cache is None:
        # Backward compatibility for Network objects deserialized from before
        # the shared cache was introduced.
        solver_cache = {}
        net._solver_cache = solver_cache

    mpc_key = solver_cache.get("base_fingerprint")
    if mpc_key is None:
        mpc_key = _network_fingerprint(net)
        solver_cache["base_fingerprint"] = mpc_key
    key = (mpc_key, _branch_limit_fingerprint(net))
    if _ACTIVE_BASE_KEY != key:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".m",
            delete=False,
        ) as temp_file:
            temp_filename = temp_file.name
        try:
            net.to_mpc(temp_filename)
            jl._gfm_init_base(temp_filename)
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        _ACTIVE_BASE_KEY = key

    bus_ids = solver_cache.get("bus_ids")
    if bus_ids is None:
        rev = solver_cache.get("reverse_bus_ids")
        if rev is None:
            rev = np.empty(net.buses.shape[0], dtype=np.int64)
            for new_idx, orig_idx in net.reverse_bus_index_mapping.items():
                rev[new_idx] = orig_idx
            solver_cache["reverse_bus_ids"] = rev
        bus_ids = rev[net.buses[:, BUS_I].astype(np.int64)]
        solver_cache["bus_ids"] = bus_ids

    ncost = solver_cache.get("ncost")
    if ncost is None:
        ncost = int(net.gencosts[0, NCOST])
        solver_cache["ncost"] = ncost
    return jl._gfm_state(
        bus_ids,
        net.buses[:, BUS_TYPE].astype(np.int64),
        net.buses[:, PD],
        net.buses[:, QD],
        net.buses[:, VM],
        net.buses[:, VA],
        net.gens[:, PG],
        net.gens[:, QG],
        net.gens[:, VG],
        net.gens[:, GEN_STATUS].astype(np.int64),
        np.ascontiguousarray(net.gencosts[:, COST : COST + ncost]),
        net.branches[:, BR_STATUS].astype(np.int64),
        net.branches[:, BR_R],
        net.branches[:, BR_X],
        net.branches[:, BR_B],
    )


def run_opf(net: Network, jl: Any) -> Dict[str, Any]:
    """Run Optimal Power Flow (OPF) calculation using Julia interface.

    Args:
        net: A Network object containing the power system model.
        jl: Julia interface object for running OPF.

    Returns:
        OPF result containing termination status and solution data.

    Raises:
        RuntimeError: If OPF fails to converge or encounters an error.
    """
    try:
        data = _julia_pm_data(net, jl)

        with solver_capture("opf"):
            result = jl.run_opf(data)

        if str(result["termination_status"]) != "LOCALLY_SOLVED":
            raise RuntimeError(f"OPF did not converge: {result['termination_status']}")

        return result

    except Exception as e:
        raise RuntimeError(f"Error running OPF: {e}")


def run_pf(net: Network, jl: Any, fast: Union[bool, None] = None) -> Dict[str, Any]:
    """Run Power Flow (PF) calculation using Julia interface.

    This function runs the power flow calculation using the Julia interface
    and returns the result with termination status.

    Args:
        net: A network object containing the power system model.
        jl: Julia interface object for running power flow.
        fast: If True, use the direct (non-optimizer) computation. If None, defaults to False (uses optimizer-based solver).

    Returns:
        Power flow result containing termination status and solution data.

    Raises:
        RuntimeError: If power flow fails to converge or encounters an error.
    """
    try:
        data = _julia_pm_data(net, jl)

        # Run PF
        with solver_capture("pf"):
            result = jl.run_pf_fast_data(data) if fast else jl.run_pf_data(data)
        if (
            fast
            and str(result["termination_status"]) != "True"
            or (not fast and str(result["termination_status"]) != "LOCALLY_SOLVED")
        ):
            raise RuntimeError(
                f"PF did not converge: {result['termination_status']}, fast={fast}",
            )

        return result

    except Exception as e:
        raise RuntimeError(f"Error running PF: {e}")


def run_dcpf(net: Network, jl: Any, fast: Union[bool, None] = None) -> Dict[str, Any]:
    """Run DC Power Flow (DCPF) calculation using Julia interface.

    This function runs the DC power flow calculation using the Julia interface
    and returns the result with termination status.

    Args:
        net: A network object containing the power system model.
        jl: Julia interface object for running DC power flow.
        fast: If True, use the direct (non-optimizer) computation. If None, defaults to False (uses optimizer-based solver).

    Returns:
        DC power flow result containing termination status and solution data.

    Raises:
        RuntimeError: If DC power flow fails to converge or encounters an error.
    """
    try:
        data = _julia_pm_data(net, jl)

        # Run DCPF (fast or standard)
        with solver_capture("dcpf"):
            result = jl.run_dcpf_fast_data(data) if fast else jl.run_dcpf_data(data)

        if (
            fast
            and str(result["termination_status"]) != "True"
            or (not fast and str(result["termination_status"]) != "LOCALLY_SOLVED")
        ):
            raise RuntimeError(
                f"DC PF did not converge: {result['termination_status']}, fast={fast}",
            )

        return result

    except Exception as e:
        raise RuntimeError(f"Error running DC PF: {e}")


def run_dcopf(net: Network, jl: Any) -> Dict[str, Any]:
    """Run DC Optimal Power Flow (DC OPF) calculation using Julia interface.

    This function runs the DC optimal power flow calculation using the Julia interface
    and returns the result with termination status.

    Args:
        net: A network object containing the power system model.
        jl: Julia interface object for running DC OPF.

    Returns:
        DC OPF result containing termination status and solution data.

    Raises:
        RuntimeError: If DC OPF fails to converge or encounters an error.
    """
    try:
        data = _julia_pm_data(net, jl)

        # Run DC OPF
        with solver_capture("dcopf"):
            result = jl.run_dcopf(data)

        if str(result["termination_status"]) != "LOCALLY_SOLVED":
            raise RuntimeError(
                f"DC OPF did not converge: {result['termination_status']}",
            )

        return result

    except Exception as e:
        raise RuntimeError(f"Error running DC OPF: {e}")


def compare_pf_results(
    net: Network,
    jl: Any,
    case_name: str,
    fast: bool,
    solver_type: str = "pf",
) -> bool:
    """Compare results from run_pf/run_opf (temp file) vs direct solve on original case file.

    This function verifies that the results from running PF/OPF on a temporary file
    (created by net.to_mpc()) are the same as running PF/OPF directly on the original
    case file from the grids directory.

    Args:
        net: A Network object containing the power system model.
        jl: Julia interface object for running power flow.
        case_name: Name of the case (e.g., 'case24_ieee_rts') to find the original file.
        solver_type: Type of solver to test - "pf" for power flow or "opf" for optimal power flow.

    Returns:
        True if results match exactly, False otherwise.

    Raises:
        RuntimeError: If either PF/OPF run fails to converge.
        FileNotFoundError: If the original case file is not found.
        ValueError: If solver_type is not "pf" or "opf".
    """
    import os

    if solver_type not in ["pf", "opf"]:
        raise ValueError(f"solver_type must be 'pf' or 'opf', got '{solver_type}'")

    solver_name = "PF" if solver_type == "pf" else "OPF"

    # Step 1: Run solver using temporary file (current method)
    print(f"Running {solver_name} on temporary file for {case_name}...")
    try:
        if solver_type == "pf":
            temp_result = run_pf(net, jl, fast)
        else:  # opf
            temp_result = run_opf(net, jl)
        temp_converged = True
    except RuntimeError as e:
        print(f"Error running {solver_name} on temporary file: {e}")
        temp_converged = False

    # Step 2: Run solver directly on original case file
    original_file_path = f"gridfm_datakit/grids/pglib_opf_{case_name}.m"

    if not os.path.exists(original_file_path):
        raise FileNotFoundError(f"Original case file not found: {original_file_path}")

    print(f"Running {solver_name} directly on original file: {original_file_path}")
    try:
        if solver_type == "pf" and fast:
            original_result = jl.run_pf_fast(original_file_path)
            original_converged = str(original_result["termination_status"]) == "True"
        elif solver_type == "pf" and not fast:
            original_result = jl.run_pf(original_file_path)
            original_converged = (
                str(original_result["termination_status"]) == "LOCALLY_SOLVED"
            )
        elif solver_type == "opf":
            original_result = jl.run_opf(original_file_path)
            original_converged = (
                str(original_result["termination_status"]) == "LOCALLY_SOLVED"
            )
    except Exception as e:
        print(f"Error running {solver_name} directly on original file: {e}")
        original_converged = False

    # Step 3: Compare results
    print(f"Comparing {solver_name} results for {case_name}...")

    if temp_converged != original_converged:
        print(
            f"Termination status mismatch: temp_converged={temp_converged}, original_converged={original_converged}",
        )
        return False

    if not temp_converged and not original_converged and solver_type == "pf":
        print(
            f"Both {solver_name} results did not converge. This is expected because the gen setpoints are not necessarily right in the case files.",
        )
        return True

    if not temp_converged and not original_converged and solver_type == "opf":
        print(
            f"Both {solver_name} results did not converge. This is not expected for OPF.",
        )
        return False

    # Compare solution data if they converged
    temp_solution = temp_result["solution"]
    original_solution = original_result["solution"]

    # Check same number and indices of buses and generators using sets
    temp_buses = temp_solution["bus"]
    original_buses = original_solution["bus"]
    temp_gens = temp_solution["gen"]
    original_gens = original_solution["gen"]

    temp_bus_ids = set(temp_buses.keys())
    original_bus_ids = set(original_buses.keys())
    temp_gen_ids = set(temp_gens.keys())
    original_gen_ids = set(original_gens.keys())

    if temp_bus_ids != original_bus_ids:
        print(
            f"Bus ID sets don't match: temp={temp_bus_ids}, original={original_bus_ids}",
        )
        return False

    if temp_gen_ids != original_gen_ids:
        print(
            f"Generator ID sets don't match: temp={temp_gen_ids}, original={original_gen_ids}",
        )
        return False

    # Compare bus voltages and angles
    for bus_id in temp_bus_ids:
        temp_vm = temp_buses[bus_id]["vm"]
        temp_va = temp_buses[bus_id]["va"]
        original_vm = original_buses[bus_id]["vm"]
        original_va = original_buses[bus_id]["va"]

        # Check if voltages and angles match exactly
        if (not np.allclose(temp_vm, original_vm)) or (
            not np.allclose(temp_va, original_va)
        ):
            print(
                f"Bus {bus_id} mismatch: temp_vm={temp_vm}, original_vm={original_vm}, temp_va={temp_va}, original_va={original_va}",
            )
            return False

    # Compare generator power outputs
    for gen_id in temp_gen_ids:
        temp_pg = temp_gens[gen_id]["pg"]
        temp_qg = temp_gens[gen_id]["qg"]
        original_pg = original_gens[gen_id]["pg"]
        original_qg = original_gens[gen_id]["qg"]

        # Check if power outputs match exactly
        if (not np.allclose(temp_pg, original_pg)) or (
            not np.allclose(temp_qg, original_qg)
        ):
            print(
                f"Gen {gen_id} mismatch: temp_pg={temp_pg}, original_pg={original_pg}, temp_qg={temp_qg}, original_qg={original_qg}",
            )
            return False

    # All checks passed
    print(f"All {len(temp_buses)} buses match exactly")
    print(f"All {len(temp_gens)} generators match exactly")
    print(f"{solver_name} results are identical for {case_name}")

    return True
