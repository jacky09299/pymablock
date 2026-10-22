import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import scqubits as scq
scq.settings.NUM_CPUS = 1
import numpy as np
from pymablock import block_diagonalize


def create_circuit_from_yaml(yaml_string, ext_basis="harmonic"):
    """
    scqubits yaml helper for cross-version compatibility.
    """
    if hasattr(scq.Circuit, "from_yaml_string"):
        return scq.Circuit.from_yaml_string(yaml_string, ext_basis=ext_basis)
    else:
        return scq.Circuit(yaml_string, from_file=False, ext_basis=ext_basis)


def multi_to_flat(multi_idx, dims):
    """
    Convert multi-subsystem bare index tuple to flat matrix index.
    """
    idx = 0
    for i, d in zip(multi_idx, dims):
        idx = idx * d + i
    return idx


def build_coupled_circuit_hilbertspace(
    L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, truncated_dim=4
):
    """
    Construct scqubits HilbertSpace for two transmons coupled to a multimode coupler network.
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

    trans_mat = np.block([[I2, Z24], [Z42, T]])

    circuit = create_circuit_from_yaml(circuit_description, ext_basis="harmonic")
    circuit.configure(transformation_matrix=trans_mat)

    periodic_indices = circuit.var_categories["periodic"]
    extended_indices = circuit.var_categories["extended"]

    coefs = circuit.hamiltonian_symbolic.as_coefficients_dict()

    import sympy as sm
    n_syms = {i: sm.Symbol(f"n{i}") for i in periodic_indices}
    Q_syms = {i: sm.Symbol(f"Q{i}") for i in extended_indices}
    theta_syms = {i: sm.Symbol(f"θ{i}") for i in extended_indices}

    EJ_map = {1: EJ1, 2: EJ2}
    qubits = {}
    for i in periodic_indices:
        EC_i = float(coefs.get(n_syms[i] ** 2, 0)) / 4.0
        EJ_i = EJ_map.get(i, EJ1)
        qubits[i] = scq.Transmon(
            EJ=EJ_i, EC=EC_i, ng=0.0, ncut=30, truncated_dim=truncated_dim
        )

    oscillators = {}
    res_E = {}
    res_l_osc = {}
    for idx in extended_indices:
        c_Q2 = float(coefs.get(Q_syms[idx] ** 2, 0))
        c_theta2 = float(coefs.get(theta_syms[idx] ** 2, 0))
        E_val = np.sqrt(abs(c_Q2 * c_theta2 * 4))
        l_osc_val = scq.core.oscillator.convert_to_l_osc(c_Q2 * 2, c_theta2 * 2)
        res_E[idx] = E_val
        res_l_osc[idx] = l_osc_val
        oscillators[idx] = scq.Oscillator(
            E_osc=E_val, l_osc=l_osc_val, truncated_dim=truncated_dim
        )

    sorted_ext_indices = sorted(extended_indices, key=lambda x: res_E[x])

    subsystems = [qubits[i] for i in periodic_indices] + [
        oscillators[idx] for idx in sorted_ext_indices
    ]
    hilbertspace = scq.HilbertSpace(subsystems)

    def get_term(op1, op2):
        return float(coefs.get(op1 * op2, 0))

    # Transmon-Transmon coupling
    for idx_a, i in enumerate(periodic_indices):
        for j in periodic_indices[idx_a + 1 :]:
            g_val = get_term(n_syms[i], n_syms[j])
            if abs(g_val) > 1e-12:
                hilbertspace.add_interaction(
                    g=g_val, op1=qubits[i].n_operator, op2=qubits[j].n_operator
                )

    # Transmon-Resonator coupling
    for i in periodic_indices:
        for idx in sorted_ext_indices:
            g_val = get_term(n_syms[i], Q_syms[idx])
            if abs(g_val) > 1e-12:
                hilbertspace.add_interaction(
                    g=g_val, op1=qubits[i].n_operator, op2=oscillators[idx].n_operator
                )

    # Resonator-Resonator coupling
    for idx_a, idx1 in enumerate(sorted_ext_indices):
        for idx2 in sorted_ext_indices[idx_a + 1 :]:
            g_val = get_term(Q_syms[idx1], Q_syms[idx2])
            if abs(g_val) > 1e-12:
                hilbertspace.add_interaction(
                    g=g_val,
                    op1=oscillators[idx1].n_operator,
                    op2=oscillators[idx2].n_operator,
                )

    return hilbertspace, subsystems


def calculate_ZZ_XX_disp_pymablock(
    L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, pert_order=4, truncated_dim=4
):
    """
    Calculate ZZ, XX coupling, qubit frequencies, coupler frequencies, and dispersive shifts
    using Pymablock quasi-degenerate matrix perturbation theory.

    Target subspace includes all 16 states with q1, q2 in {0, 1} and at most 1 photon in all resonators.

    Returns:
    --------
    fq1, fq2, fc1, fc2, ZZ, XX, disp_10_3, disp_01_3, disp_10_4, disp_01_4
    (all frequencies/coupling values in GHz, matching ZZXXdisp_cal.py output signature).
    """
    hilbertspace, subsystems = build_coupled_circuit_hilbertspace(
        L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, truncated_dim=truncated_dim
    )

    dims = [sub.truncated_dim for sub in subsystems]

    H0_mat = hilbertspace.bare_hamiltonian().full()
    H_full = hilbertspace.hamiltonian().full()
    V_mat = H_full - H0_mat

    dim = H0_mat.shape[0]

    # Include all 16 target states with q1, q2 in {0, 1} and r1 + r2 + r3 <= 1
    states_to_include = []
    for q1 in range(2):
        for q2 in range(2):
            for r1 in range(2):
                for r2 in range(2):
                    for r3 in range(2):
                        if r1 + r2 + r3 <= 1:
                            states_to_include.append((q1, q2, r1, r2, r3))

    subset_flat = [multi_to_flat(idx, dims) for idx in states_to_include]

    vecs_Sub = np.zeros((dim, len(subset_flat)))
    for col, idx in enumerate(subset_flat):
        vecs_Sub[idx, col] = 1.0

    other_sub = sorted(list(set(range(dim)) - set(subset_flat)))
    vecs_Rest = np.zeros((dim, len(other_sub)))
    for col, idx in enumerate(other_sub):
        vecs_Rest[idx, col] = 1.0

    # Call Pymablock matrix block-diagonalization
    H_tilde, U, S = block_diagonalize(
        [H0_mat, V_mat], subspace_eigenvectors=[vecs_Sub, vecs_Rest]
    )

    # Sum effective Hamiltonian corrections up to requested order
    H_eff_sub = np.zeros_like(H_tilde[0, 0, 0], dtype=complex)
    for n in range(pert_order + 1):
        H_eff_sub += H_tilde[0, 0, n]

    # Extract state energies from diagonal of effective Hamiltonian
    E = {}
    for name, idx in zip(states_to_include, subset_flat):
        sub_col = subset_flat.index(idx)
        E[name] = np.real(H_eff_sub[sub_col, sub_col])

    fq1 = E[(1, 0, 0, 0, 0)] - E[(0, 0, 0, 0, 0)]
    fq2 = E[(0, 1, 0, 0, 0)] - E[(0, 0, 0, 0, 0)]
    fc1 = E[(0, 0, 1, 0, 0)] - E[(0, 0, 0, 0, 0)]
    fc2 = E[(0, 0, 0, 1, 0)] - E[(0, 0, 0, 0, 0)]

    # ZZ coupling
    ZZ = E[(1, 1, 0, 0, 0)] - E[(1, 0, 0, 0, 0)] - E[(0, 1, 0, 0, 0)] + E[(0, 0, 0, 0, 0)]

    # XX coupling (off-diagonal exchange matrix element between |10000> and |01000>)
    idx_10 = subset_flat.index(multi_to_flat((1, 0, 0, 0, 0), dims))
    idx_01 = subset_flat.index(multi_to_flat((0, 1, 0, 0, 0), dims))
    XX = np.real(H_eff_sub[idx_10, idx_01])

    # Dispersive shifts
    disp_10_3 = (E[(0, 0, 1, 0, 0)] - E[(0, 0, 0, 0, 0)]) - (
        E[(1, 0, 1, 0, 0)] - E[(1, 0, 0, 0, 0)]
    )
    disp_01_3 = (E[(0, 0, 1, 0, 0)] - E[(0, 0, 0, 0, 0)]) - (
        E[(0, 1, 1, 0, 0)] - E[(0, 1, 0, 0, 0)]
    )

    disp_10_4 = (E[(0, 0, 0, 1, 0)] - E[(0, 0, 0, 0, 0)]) - (
        E[(1, 0, 0, 1, 0)] - E[(1, 0, 0, 0, 0)]
    )
    disp_01_4 = (E[(0, 0, 0, 1, 0)] - E[(0, 0, 0, 0, 0)]) - (
        E[(0, 1, 0, 1, 0)] - E[(0, 1, 0, 0, 0)]
    )

    return fq1, fq2, fc1, fc2, ZZ, XX, disp_10_3, disp_01_3, disp_10_4, disp_01_4


if __name__ == "__main__":
    L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2 = (
        2,
        114,
        47.98418086298829,
        5.0,
        6.495487626697703,
        0.14260240302950591,
        20,
        0.14260240302950591,
        30,
    )

    fq1, fq2, fc1, fc2, ZZ, XX, disp_10_3, disp_01_3, disp_10_4, disp_01_4 = (
        calculate_ZZ_XX_disp_pymablock(
            L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, pert_order=4
        )
    )

    print(f"ZZ: {ZZ * 1000:.6g} MHz")
    print(f"XX: {XX * 1000:.6g} MHz")
    print(f"fq1: {fq1:.3g} GHz")
    print(f"fq2: {fq2:.3g} GHz")
    print(f"fc1: {fc1:.3g} GHz")
    print(f"fc2: {fc2:.3g} GHz")
    print(f"disp_10_3: {disp_10_3 * 1000:.4g} MHz")
    print(f"disp_01_3: {disp_01_3 * 1000:.4g} MHz")
    print(f"disp_10_4: {disp_10_4 * 1000:.4g} MHz")
    print(f"disp_01_4: {disp_01_4 * 1000:.4g} MHz")
