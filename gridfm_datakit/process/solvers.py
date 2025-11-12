"""
Power flow and optimal power flow solvers using Julia interface.

This module provides functions for running power flow (PF), optimal power flow (OPF),
DC power flow (DCPF), and DC optimal power flow (DC OPF) calculations using Julia's PowerModels.jl package.
It also includes functionality for comparing results between temporary files
and original case files.
"""

import numpy as np
import tempfile
import os
from typing import Any, Dict
from gridfm_datakit.network import Network
from typing import Union


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
    # Create a temporary file for the MATPOWER case
    with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        # Save network to temporary file
        net.to_mpc(temp_filename)

        result = jl.run_opf(temp_filename)

        if str(result["termination_status"]) != "LOCALLY_SOLVED":
            raise RuntimeError(f"OPF did not converge: {result['termination_status']}")

        return result

    except Exception as e:
        raise RuntimeError(f"Error running OPF: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
    # TODO: try warm start


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

    # Create a temporary file for the MATPOWER case
    with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        # Save network to temporary file
        net.to_mpc(temp_filename)

        # Run PF
        result = jl.run_pf_fast(temp_filename) if fast else jl.run_pf(temp_filename)
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
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


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

    # Create a temporary file for the MATPOWER case
    with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        # Save network to temporary file
        net.to_mpc(temp_filename)

        # Run DCPF (fast or standard)
        result = jl.run_dcpf_fast(temp_filename) if fast else jl.run_dcpf(temp_filename)

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
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


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
    # Create a temporary file for the MATPOWER case
    with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        # Save network to temporary file
        net.to_mpc(temp_filename)

        # Run DC OPF
        result = jl.run_dcopf(temp_filename)

        if str(result["termination_status"]) != "LOCALLY_SOLVED":
            raise RuntimeError(
                f"DC OPF did not converge: {result['termination_status']}",
            )

        return result

    except Exception as e:
        raise RuntimeError(f"Error running DC OPF: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


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
