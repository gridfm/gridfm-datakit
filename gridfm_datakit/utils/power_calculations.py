import numpy as np
import pyomo.environ as pyo


def Pi(vm, va, G, B, i):
    """
    Compute active power injection at bus i.

    Args:
        vm: voltage magnitudes (numpy array or Pyomo variables)
        va: voltage angles (numpy array or Pyomo variables)
        G: real part of Y-bus matrix
        B: imaginary part of Y-bus matrix
        i: bus index

    Returns:
        Active power injection at bus i
    """
    expr = 0
    if isinstance(vm, np.ndarray):
        for j in range(len(vm)):
            if G[i, j] != 0 or B[i, j] != 0:
                theta_ij = va[i] - va[j]
                expr += (
                    vm[i]
                    * vm[j]
                    * (G[i, j] * np.cos(theta_ij) + B[i, j] * np.sin(theta_ij))
                )
    else:
        for j in range(len(vm)):
            if G[i, j] != 0 or B[i, j] != 0:
                theta_ij = va[i] - va[j]
                expr += (
                    vm[i]
                    * vm[j]
                    * (G[i, j] * pyo.cos(theta_ij) + B[i, j] * pyo.sin(theta_ij))
                )
    return expr


def Qi(vm, va, G, B, i):
    """
    Compute reactive power injection at bus i.

    Args:
        vm: voltage magnitudes (numpy array or Pyomo variables)
        va: voltage angles (numpy array or Pyomo variables)
        G: real part of Y-bus matrix
        B: imaginary part of Y-bus matrix
        i: bus index

    Returns:
        Reactive power injection at bus i
    """
    expr = 0
    if isinstance(vm, np.ndarray):
        for j in range(len(vm)):
            if G[i, j] != 0 or B[i, j] != 0:
                theta_ij = va[i] - va[j]
                expr += (
                    vm[i]
                    * vm[j]
                    * (G[i, j] * np.sin(theta_ij) - B[i, j] * np.cos(theta_ij))
                )
    else:
        for j in range(len(vm)):
            if G[i, j] != 0 or B[i, j] != 0:
                theta_ij = va[i] - va[j]
                expr += (
                    vm[i]
                    * vm[j]
                    * (G[i, j] * pyo.sin(theta_ij) - B[i, j] * pyo.cos(theta_ij))
                )
    return expr


def Pij(vm, va, G, B, i, j):
    """
    Compute active power flow from bus i to bus j.

    Args:
        vm: voltage magnitudes (numpy array or Pyomo variables)
        va: voltage angles (numpy array or Pyomo variables)
        G: real part of Y-bus matrix
        B: imaginary part of Y-bus matrix
        i: from bus index
        j: to bus index

    Returns:
        Active power flow from bus i to bus j
    """
    theta_ij = va[i] - va[j]
    if isinstance(vm, np.ndarray):
        return vm[i] ** 2 * G[i, j] - vm[i] * vm[j] * (
            G[i, j] * np.cos(theta_ij) + B[i, j] * np.sin(theta_ij)
        )
    else:
        return vm[i] ** 2 * G[i, j] - vm[i] * vm[j] * (
            G[i, j] * pyo.cos(theta_ij) + B[i, j] * pyo.sin(theta_ij)
        )


def Qij(vm, va, G, B, i, j):
    """
    Compute reactive power flow from bus i to bus j.

    Args:
        vm: voltage magnitudes (numpy array or Pyomo variables)
        va: voltage angles (numpy array or Pyomo variables)
        G: real part of Y-bus matrix
        B: imaginary part of Y-bus matrix
        i: from bus index
        j: to bus index

    Returns:
        Reactive power flow from bus i to bus j
    """
    theta_ij = va[i] - va[j]
    if isinstance(vm, np.ndarray):
        return -(
            vm[i] ** 2 * B[i, j]
            + vm[i] * vm[j] * (G[i, j] * np.sin(theta_ij) - B[i, j] * np.cos(theta_ij))
        )
    else:
        return -(
            vm[i] ** 2 * B[i, j]
            + vm[i]
            * vm[j]
            * (G[i, j] * pyo.sin(theta_ij) - B[i, j] * pyo.cos(theta_ij))
        )


def get_flows(vm, va, G, B, debug=False):
    # !!! assert everything is numpy array as hadamard product creates a mess with scipy sparse matrices
    assert isinstance(vm, np.ndarray)
    assert isinstance(va, np.ndarray)
    assert isinstance(G, np.ndarray)
    assert isinstance(B, np.ndarray)

    vm = vm.reshape(-1, 1)  # shape (n, 1)
    va = va.reshape(-1, 1)  # shape (n, 1)

    theta_diff = va - va.T  # θ_i - θ_j, shape (n, n)
    vm_outer = vm @ vm.T  # V_i * V_j, shape (n, n)
    vm_sq = vm**2  # V_i^2, shape (n, 1)
    vm_sq_expanded = vm_sq @ np.ones((1, vm.shape[0]))  # shape (n, n)

    # P_ij
    V_i_2_times_G = G * vm_sq_expanded  # V_i^2 * G_ij (broadcasted across rows)
    G_times_cos_theta_diff = G * np.cos(theta_diff)
    B_times_sin_theta_diff = B * np.sin(theta_diff)
    second_term = (G_times_cos_theta_diff + B_times_sin_theta_diff) * vm_outer

    P_ij = V_i_2_times_G - second_term

    # Q_ij
    V_i_2_times_B = B * vm_sq_expanded  # V_i^2 * B_ij (broadcasted across rows)
    G_times_sin_theta_diff = G * np.sin(theta_diff)
    B_times_cos_theta_diff = B * np.cos(theta_diff)
    second_term = (G_times_sin_theta_diff - B_times_cos_theta_diff) * vm_outer

    Q_ij = -(V_i_2_times_B + second_term)

    if debug:
        print("checking P_ij and Q_ij")
        # compute P_ij with the function Pij from power_calculations.py
        # compare P_ij and P_ij_func for all elements
        for i in range(P_ij.shape[0]):
            for j in range(P_ij.shape[1]):
                P_ij_func = Pij(vm, va, G, B, i, j)
                Q_ij_func = Qij(vm, va, G, B, i, j)
                assert np.isclose(P_ij[i, j], P_ij_func)
                assert np.isclose(Q_ij[i, j], Q_ij_func)
    return P_ij, Q_ij


def get_injections(vm, va, G, B, debug=False):
    V = vm * np.exp(1j * va)
    Y = G + 1j * B
    V_conj = np.conj(V)
    Y_conj = np.conj(Y)
    S = V * (Y_conj @ V_conj)  # Element-wise multiplication

    if debug:
        print("checking get_injections")
        # compute P_ij with the function Pij from power_calculations.py
        # compare P_ij and P_ij_func for all elements
        for i in range(vm.shape[0]):
            P_func = Pi(vm, va, G, B, i)
            Q_func = Qi(vm, va, G, B, i)
            assert np.isclose(S[i].real, P_func)
            assert np.isclose(S[i].imag, Q_func)

    return S.real, S.imag
