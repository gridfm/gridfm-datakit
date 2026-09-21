"""
lightsim2grid power flow results preprocessing module.

Format lightsim2grid power flow results into PowerModels' power flow format
(per-unit, radians), the same layout the PowSyBl integration produces, so that
the existing ``pf_post_processing`` can consume them.
"""

import time
from typing import Any, Dict

import numpy as np

from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_bus import BUS_I

from .convert import initial_voltage
from .mapping import MappingL2G


def run_ls_pf(
    ls_net: Any,
    net: Network,
    mapping_l2g: MappingL2G,
    dc: bool = False,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """Run an AC (or DC) power flow with lightsim2grid and format the results.

    The power flow starts from the voltages stored in ``net.buses`` (see
    :func:`gridfm_datakit.lightsim2grid.convert.initial_voltage`).

    Args:
        ls_net: The lightsim2grid LSGrid, in sync with ``net``.
        net: The network the LSGrid was built from or updated with.
        mapping_l2g: Index maps between ``net`` and ``ls_net``.
        dc: Run a DC power flow instead of an AC one.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.

    Returns:
        The power flow results in PowerModels' format (see :func:`get_pf_res`), with the
        solving time in ``"solve_time"``.

    Raises:
        ValueError: If the power flow did not converge.
    """
    v_init = initial_voltage(net)
    start_time = time.perf_counter()
    v = (ls_net.dc_pf if dc else ls_net.ac_pf)(v_init, max_iter, tol)
    solve_time = time.perf_counter() - start_time
    return get_pf_res(ls_net, v, solve_time, net, mapping_l2g)


def get_pf_res(
    ls_net: Any,
    v: np.ndarray,
    solve_time: float,
    net: Network,
    mapping_l2g: MappingL2G,
) -> Dict[Any, Any]:
    """Format lightsim2grid power flow results for the pf_post_process function.

    Args:
        ls_net: lightsim2grid LSGrid, on which a power flow was just run
        v: complex voltage returned by the power flow (empty if it diverged)
        solve_time: power flow solving time
        net: gridfm Network the LSGrid was built from
        mapping_l2g: lightsim2grid-to-gridfm index maps

    Returns:
        Power flow results in a nested Dict format, similar to PowerModel's power flow results
    """
    if v.shape[0] == 0:
        raise ValueError(
            "Power flow computation failed: lightsim2grid did not converge",
        )

    base_mva = float(net.baseMVA)
    n_branches = net.branches.shape[0]

    # Branch flows in pu, (n_branches, 4) as pf, qf, pt, qt. Powerline
    # and transformer results are scattered back to their branch rows.
    flows = np.zeros((n_branches, 4))
    for rows, res1, res2 in (
        (mapping_l2g.line_rows, ls_net.get_line_res1(), ls_net.get_line_res2()),
        (mapping_l2g.trafo_rows, ls_net.get_trafo_res1(), ls_net.get_trafo_res2()),
    ):
        flows[rows, 0] = res1[0]
        flows[rows, 1] = res1[1]
        flows[rows, 2] = res2[0]
        flows[rows, 3] = res2[1]
    flows /= base_mva

    gen_p, gen_q, _ = ls_net.get_gen_res()
    gen_p = np.asarray(gen_p) / base_mva
    gen_q = np.asarray(gen_q) / base_mva

    reverse = net.reverse_bus_index_mapping
    vm, va = np.abs(v), np.angle(v)

    return {
        "solution": {
            "baseMVA": base_mva,
            "gen": {
                str(int(i) + 1): {"pg": gen_p[i], "qg": gen_q[i]}
                for i in net.idx_gens_in_service
            },
            "branch": {
                str(i + 1): {
                    "pf": flows[i, 0],
                    "qf": flows[i, 1],
                    "pt": flows[i, 2],
                    "qt": flows[i, 3],
                }
                for i in range(n_branches)
            },
            "bus": {
                str(reverse[int(net.buses[r, BUS_I])]): {"vm": vm[r], "va": va[r]}
                for r in range(net.buses.shape[0])
            },
            "per_unit": True,
            "pf": True,
        },
        "solve_time": solve_time,
    }
