import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import scipy.sparse as sp
import scqubits as scq
scq.settings.NUM_CPUS = 1

# Ensure compatibility between modern scipy.sparse (csc_array) and scqubits
import scqubits.utils.spectrum_utils as su
import scqubits.core.hilbert_space as hs

_orig_convert = su.convert_operator_to_qobj

def _convert_operator_to_qobj_patched(operator, subsystem, op_in_eigenbasis, evecs):
    if isinstance(operator, (sp.csc_array, sp.csr_array, sp.coo_array, sp.bsr_array, sp.dia_array, sp.sparray)):
        operator = sp.csc_matrix(operator)
    return su.convert_matrix_to_qobj(operator, subsystem, op_in_eigenbasis, evecs)

su.convert_operator_to_qobj = _convert_operator_to_qobj_patched
hs.spec_utils.convert_operator_to_qobj = _convert_operator_to_qobj_patched

import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, h
from lcapy import Circuit
from pymablock import block_diagonalize


def create_circuit_from_yaml(yaml_string, ext_basis="harmonic"):
    """
    scqubits helper for circuit creation from YAML string.
    """
    if hasattr(scq.Circuit, "from_yaml_string"):
        return scq.Circuit.from_yaml_string(yaml_string, ext_basis=ext_basis)
    else:
        return scq.Circuit(yaml_string, from_file=False, ext_basis=ext_basis)


def J12(L, C, L_side, C_side, Cc, EC1, EC2, f1_GHz, f2_GHz):
    """
    Calculates 2-port admittance / impedance based classical J12 coupling (GHz).
    """
    values = {
        "L": L,
        "C": C,
        "L_side": L_side,
        "C_side": C_side,
        "Cc": Cc,
        "EC1": EC1,
        "EC2": EC2,
        "f1_GHz": f1_GHz,
        "f2_GHz": f2_GHz,
    }

    invalid = [
        name
        for name, value in values.items()
        if not np.isfinite(value) or value <= 0
    ]
    if invalid:
        raise ValueError(f"Parameters must be finite and > 0: {invalid}")

    EC1_J = h * EC1 * 1e9
    EC2_J = h * EC2 * 1e9

    Cq1 = e**2 / (2 * EC1_J) * 1e15  # fF
    Cq2 = e**2 / (2 * EC2_J) * 1e15  # fF

    netlist = f"""
C1 6 0 {C}f; down
L1 5 6 {L}n; down

Cside3 3 5 {C_side}f; right
Lside3 3 0 {L_side}n; down
Cside4 4 5 {C_side}f; left
Lside4 4 0 {L_side}n; down
Cq1 1 0 {Cq1}f; down
Cc1 1 3 {Cc}f; right
Cq2 2 0 {Cq2}f; down
Cc2 4 2 {Cc}f; right
"""

    cct = Circuit(netlist)
    two_port = cct.twoport("1", "0", "2", "0")

    Z_sym = sm.Matrix(two_port.Zparams)

    s_candidates = [sym for sym in Z_sym.free_symbols if str(sym) == "s"]
    if len(s_candidates) != 1:
        raise ValueError(f"Unable to uniquely identify s symbol: {s_candidates}")

    s = s_candidates[0]
    Z12_expr = sm.simplify(Z_sym[0, 1])
    Z12_func = sm.lambdify(s, Z12_expr, modules="numpy")

    w1 = 2 * np.pi * f1_GHz * 1e9
    w2 = 2 * np.pi * f2_GHz * 1e9

    Cq1_SI = Cq1 * 1e-15
    Cq2_SI = Cq2 * 1e-15

    Lq1 = 1 / (Cq1_SI * w1**2)
    Lq2 = 1 / (Cq2_SI * w2**2)

    z12_w1 = complex(np.asarray(Z12_func(1j * w1)).item())
    z12_w2 = complex(np.asarray(Z12_func(1j * w2)).item())

    J12_rad_s = -0.25 * np.sqrt(
        (w1 * w2) / (Lq1 * Lq2)
    ) * np.imag(
        z12_w1 / w1 + z12_w2 / w2
    )

    return J12_rad_s / (2 * np.pi * 1e9)


def disp_ZZ_pymablock(L, C, L_side=None, C_side=None, Cc=5, EC1=0.2, EJ1=20, EC2=0.2, EJ2=20, order=6):
    """
    Computes qubit frequencies, coupler frequencies, ZZ coupling, XX exchange coupling,
    and dispersive shifts using Pymablock quasi-degenerate perturbation theory.

    Parameters
    ----------
    L, C, L_side, C_side, Cc : float
        Circuit component values.
    EC1, EJ1, EC2, EJ2 : float
        Transmon qubit 1 and qubit 2 parameters in GHz.
    order : int
        Maximum perturbation order for Pymablock block diagonalization (default: 6).

    Returns
    -------
    tuple
        (fq1, fq2, fc1, fc2, ZZ, XX, disp_10_3, disp_01_3, disp_10_4, disp_01_4)
        Energies/frequencies in GHz except ZZ, XX, disp in GHz.
    """
    resonator_description = f"""
    branches:
    - [C,4,0,C_res={C} fF]
    - [L,3,4,L_res={L} nH]
    - [C,1,3,C_res_left={C_side} fF]
    - [L,1,0,L_res_left={L_side} nH]
    - [C,2,3,C_res_right={C_side} fF]
    - [L,2,0,L_res_right={L_side} nH]
    """

    circuit_description = f"""
    branches:
    - [C,6,0,C_res={C} fF]
    - [L,5,6,L_res={L} nH]
    - [C,3,5,C_res_left={C_side} fF]
    - [L,3,0,L_res_left={L_side} nH]
    - [C,4,5,C_res_right={C_side} fF]
    - [L,4,0,L_res_right={L_side} nH]
    - [C,1,0,EC1={EC1} GHz]
    - [JJ,1,0,EJ1={EJ1},0.000000001 fF]
    - [C,1,3,Cc = {Cc} fF]
    - [JJ,2,0,EJ2={EJ2},0.000000001 fF]
    - [C,2,0,EC2={EC2} GHz]
    - [C,4,2,Cc]
    """

    resonator = create_circuit_from_yaml(resonator_description, ext_basis="harmonic")
    T = resonator.transformation_matrix

    I2 = np.eye(2)
    Z24 = np.zeros((2, 4))
    Z42 = np.zeros((4, 2))

    trans_mat = np.block([
        [I2,  Z24],
        [Z42, T]
    ])

    circuit = create_circuit_from_yaml(circuit_description, ext_basis="harmonic")
    circuit.configure(transformation_matrix=trans_mat)

    periodic_indices = circuit.var_categories["periodic"]
    extended_indices = circuit.var_categories["extended"]

    coefs = circuit.hamiltonian_symbolic.as_coefficients_dict()

    n_syms = {i: sm.Symbol(f"n{i}") for i in periodic_indices}
    Q_syms = {i: sm.Symbol(f"Q{i}") for i in extended_indices}

    EJ_map = {1: EJ1, 2: EJ2}
    qubits = {}
    for i in periodic_indices:
        EC_i = float(coefs.get(n_syms[i]**2, 0)) / 4.0
        EJ_i = EJ_map.get(i, EJ1)
        qubits[i] = scq.Transmon(EJ=EJ_i, EC=EC_i, ng=0.0, ncut=30, truncated_dim=4)

    oscillators = {}
    res_E = {}
    res_l_osc = {}
    for idx in extended_indices:
        c_Q2 = float(coefs.get(Q_syms[idx]**2, 0))
        c_theta2 = float(coefs.get(sm.Symbol(f"θ{idx}")**2, 0))
        E_val = np.sqrt(abs(c_Q2 * c_theta2 * 4))
        l_osc_val = scq.core.oscillator.convert_to_l_osc(c_Q2 * 2, c_theta2 * 2)
        res_E[idx] = E_val
        res_l_osc[idx] = l_osc_val
        oscillators[idx] = scq.Oscillator(E_osc=E_val, l_osc=l_osc_val, truncated_dim=4)

    sorted_ext_indices = sorted(extended_indices, key=lambda x: res_E[x])
    subsystems = [qubits[i] for i in periodic_indices] + [oscillators[idx] for idx in sorted_ext_indices]

    def get_term(op1, op2):
        return float(coefs.get(op1 * op2, 0))

    hs_osc = scq.HilbertSpace(subsystems)
    hs_qubit = scq.HilbertSpace(subsystems)

    # Transmon-Transmon and Transmon-Resonator couplings
    for idx_a, i in enumerate(periodic_indices):
        for j in periodic_indices[idx_a + 1:]:
            g_val = get_term(n_syms[i], n_syms[j])
            if abs(g_val) > 1e-12:
                hs_qubit.add_interaction(g=g_val, op1=qubits[i].n_operator, op2=qubits[j].n_operator)

    for i in periodic_indices:
        for idx in sorted_ext_indices:
            g_val = get_term(n_syms[i], Q_syms[idx])
            if abs(g_val) > 1e-12:
                hs_qubit.add_interaction(g=g_val, op1=qubits[i].n_operator, op2=oscillators[idx].n_operator)

    # Resonator-Resonator network couplings (included in H0 to form exact normal modes)
    for idx_a, idx1 in enumerate(sorted_ext_indices):
        for idx2 in sorted_ext_indices[idx_a + 1:]:
            g_val = get_term(Q_syms[idx1], Q_syms[idx2])
            if abs(g_val) > 1e-12:
                hs_osc.add_interaction(g=g_val, op1=oscillators[idx1].n_operator, op2=oscillators[idx2].n_operator)

    bare_esys = {sub: sub.eigensys(evals_count=sub.truncated_dim) for sub in subsystems}

    H_bare = hs_qubit.bare_hamiltonian(bare_esys=bare_esys).full()
    V_osc = hs_osc.interaction_hamiltonian(bare_esys=bare_esys).full()
    V_qubit = hs_qubit.interaction_hamiltonian(bare_esys=bare_esys).full()

    # Define unperturbed H0 including full linear coupler network
    H0_full = H_bare + V_osc
    V_pert = V_qubit

    # Diagonalize H0_full to establish exact unperturbed eigenbasis
    e_H0, v_H0 = np.linalg.eigh(H0_full)
    H0_diag = np.diag(e_H0)
    V_trans = v_H0.T.conj() @ V_pert @ v_H0

    dims = [sub.truncated_dim for sub in subsystems]

    def bare_tuple_to_idx(t):
        res = 0
        for val, d in zip(t, dims):
            res = res * d + val
        return res

    # Define target subspace states: 0-photon qubit manifold and 1-photon resonator manifolds
    states_00 = [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (1, 1, 0, 0, 0)]
    states_r1 = [(0, 0, 1, 0, 0), (1, 0, 1, 0, 0), (0, 1, 1, 0, 0), (1, 1, 1, 0, 0)]
    states_r2 = [(0, 0, 0, 1, 0), (1, 0, 0, 1, 0), (0, 1, 0, 1, 0), (1, 1, 0, 1, 0)]

    all_target_states = states_00 + states_r1 + states_r2
    idx_target_H0 = []
    for st in all_target_states:
        bare_idx = bare_tuple_to_idx(st)
        overlaps = np.abs(v_H0[bare_idx, :])**2
        idx_target_H0.append(np.argmax(overlaps))

    dim = H0_diag.shape[0]
    vecs_A = np.zeros((dim, len(all_target_states)))
    for col, idx in enumerate(idx_target_H0):
        vecs_A[idx, col] = 1.0

    other_idx = [i for i in range(dim) if i not in idx_target_H0]
    vecs_B = np.zeros((dim, len(other_idx)))
    for col, idx in enumerate(other_idx):
        vecs_B[idx, col] = 1.0

    # Perform Pymablock quasi-degenerate block-diagonalization
    H_tilde, U, S = block_diagonalize([H0_diag, V_trans], subspace_eigenvectors=[vecs_A, vecs_B])

    # Sum corrections up to order
    H_eff = sum(H_tilde[0, 0, k] for k in range(order + 1))
    evals, evecs = np.linalg.eigh(H_eff)

    matched_evals = {}
    for col_idx, st in enumerate(all_target_states):
        overlaps = np.abs(evecs[col_idx, :])**2
        matched_evals[st] = evals[np.argmax(overlaps)]

    E_00000 = matched_evals[(0, 0, 0, 0, 0)]
    E_10000 = matched_evals[(1, 0, 0, 0, 0)]
    E_01000 = matched_evals[(0, 1, 0, 0, 0)]
    E_11000 = matched_evals[(1, 1, 0, 0, 0)]

    E_00100 = matched_evals[(0, 0, 1, 0, 0)]
    E_10100 = matched_evals[(1, 0, 1, 0, 0)]
    E_01100 = matched_evals[(0, 1, 1, 0, 0)]

    E_00010 = matched_evals[(0, 0, 0, 1, 0)]
    E_10010 = matched_evals[(1, 0, 0, 1, 0)]
    E_01010 = matched_evals[(0, 1, 0, 1, 0)]

    fq1 = E_10000 - E_00000
    fq2 = E_01000 - E_00000
    fc1 = E_00100 - E_00000
    fc2 = E_00010 - E_00000

    ZZ = E_11000 - E_01000 - E_10000 + E_00000

    # XX exchange coupling from off-diagonal term between |1,0> and |0,1> in H_eff
    XX = np.abs(H_eff[1, 2])

    disp_10_3 = (E_00100 - E_00000) - (E_10100 - E_10000)
    disp_01_3 = (E_00100 - E_00000) - (E_01100 - E_01000)

    disp_10_4 = (E_00010 - E_00000) - (E_10010 - E_10000)
    disp_01_4 = (E_00010 - E_00000) - (E_01010 - E_01000)

    return fq1, fq2, fc1, fc2, ZZ, XX, disp_10_3, disp_01_3, disp_10_4, disp_01_4


if __name__ == "__main__":
    L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2 = (
        2, 114, 47.98418086298829, 5.0, 6.495487626697703,
        0.14260240302950591, 20, 0.14260240302950591, 30
    )

    fq1, fq2, fc1, fc2, ZZ, XX, disp_10_3, disp_01_3, disp_10_4, disp_01_4 = disp_ZZ_pymablock(
        L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, order=6
    )

    print("--- Pymablock Perturbation Theory Calculation ---")
    print(f"fq1: {fq1:.4g} GHz")
    print(f"fq2: {fq2:.4g} GHz")
    print(f"fc1: {fc1:.4g} GHz")
    print(f"fc2: {fc2:.4g} GHz")
    print(f"ZZ: {ZZ*1000:.6g} MHz")
    print(f"XX: {XX*1000:.6g} MHz")
    print(f"disp_10_3: {disp_10_3*1000:.4g} MHz")
    print(f"disp_01_3: {disp_01_3*1000:.4g} MHz")
    print(f"disp_10_4: {disp_10_4*1000:.4g} MHz")
    print(f"disp_01_4: {disp_01_4*1000:.4g} MHz")
