import scipy.sparse as sp
import scqubits.utils.spectrum_utils as su
import scqubits.core.hilbert_space as hs

# Apply compatibility patch before importing ZZXXdisp_cal
def _convert_operator_to_qobj_patched(operator, subsystem, op_in_eigenbasis, evecs):
    if isinstance(operator, (sp.csc_array, sp.csr_array, sp.coo_array, sp.bsr_array, sp.dia_array, sp.sparray)):
        operator = sp.csc_matrix(operator)
    return su.convert_matrix_to_qobj(operator, subsystem, op_in_eigenbasis, evecs)

su.convert_operator_to_qobj = _convert_operator_to_qobj_patched
hs.spec_utils.convert_operator_to_qobj = _convert_operator_to_qobj_patched

import numpy as np
import pytest
from ZZXXdisp_cal import disp_ZZ
from pymablock_ZZXXdisp import disp_ZZ_pymablock


def test_comparison_ed_vs_pymablock():
    L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2 = (
        2, 114, 47.98418086298829, 5.0, 6.495487626697703,
        0.14260240302950591, 20, 0.14260240302950591, 30
    )

    # 1. Exact Diagonalization / Original Calculation
    ed_results = disp_ZZ(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2)
    fq1_ed, fq2_ed, fc1_ed, fc2_ed, ZZ_ed, XX_ed, d10_3_ed, d01_3_ed, d10_4_ed, d01_4_ed = ed_results

    print("\n=========================================================================")
    print("           COMPARISON TABLE: EXACT DIAG (ED) VS PYMABLOCK                ")
    print("=========================================================================")
    print(f" {'Quantity':<12} | {'ED / Baseline':<15} | {'Pymablock (O=6)':<15} | {'Pymablock (O=10)':<15} | {'Diff (O=10)':<12}")
    print("-" * 78)

    # 2. Pymablock at Order 6 and Order 10
    pm6_results = disp_ZZ_pymablock(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, order=6)
    pm10_results = disp_ZZ_pymablock(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, order=10)

    fq1_p10, fq2_p10, fc1_p10, fc2_p10, ZZ_p10, XX_p10, d10_3_p10, d01_3_p10, d10_4_p10, d01_4_p10 = pm10_results
    fq1_p6, fq2_p6, fc1_p6, fc2_p6, ZZ_p6, XX_p6, d10_3_p6, d01_3_p6, d10_4_p6, d01_4_p6 = pm6_results

    labels = ["fq1 (GHz)", "fq2 (GHz)", "fc1 (GHz)", "fc2 (GHz)", "ZZ (MHz)", "XX (MHz)", "disp_10_3 (MHz)", "disp_01_3 (MHz)", "disp_10_4 (MHz)", "disp_01_4 (MHz)"]
    scale = [1, 1, 1, 1, 1000, 1000, 1000, 1000, 1000, 1000]

    for label, s, ed_v, p6_v, p10_v in zip(labels, scale, ed_results, pm6_results, pm10_results):
        ed_scaled = ed_v * s
        p6_scaled = p6_v * s
        p10_scaled = p10_v * s
        diff = p10_scaled - ed_scaled
        print(f" {label:<12} | {ed_scaled:15.6f} | {p6_scaled:15.6f} | {p10_scaled:15.6f} | {diff:12.6f}")

    print("=========================================================================\n")

    # Assertions to verify convergence and agreement with ED for dispersive shifts and frequencies
    np.testing.assert_allclose(fq1_p10, fq1_ed, rtol=1e-3)
    np.testing.assert_allclose(fq2_p10, fq2_ed, rtol=1e-3)
    np.testing.assert_allclose(fc1_p10, fc1_ed, rtol=1e-3)
    np.testing.assert_allclose(fc2_p10, fc2_ed, rtol=1e-3)

    # Dispersive shifts agreement (< 5% error at order 10)
    np.testing.assert_allclose(d10_4_p10 * 1000, d10_4_ed * 1000, rtol=0.05)
    np.testing.assert_allclose(d01_4_p10 * 1000, d01_4_ed * 1000, rtol=0.05)


if __name__ == "__main__":
    test_comparison_ed_vs_pymablock()
