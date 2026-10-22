import numpy as np
from ZZXXdisp_cal import disp_ZZ as disp_ZZ_exact
from pymablock_ZZXXdisp_cal import calculate_ZZ_XX_disp_pymablock


def test_compare_pymablock_and_exact_diagonalization():
    """
    Compare ZZ, XX, and dispersive shifts calculated by Pymablock perturbation theory
    vs exact diagonalization in ZZXXdisp_cal.py.
    """
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

    # 1. Exact Diagonalization & Black-Box Impedance results from original script
    fq1_e, fq2_e, fc1_e, fc2_e, ZZ_e, XX_e, d10_3_e, d01_3_e, d10_4_e, d01_4_e = (
        disp_ZZ_exact(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2)
    )

    # 2. Pymablock Perturbation Theory results
    fq1_p, fq2_p, fc1_p, fc2_p, ZZ_p, XX_p, d10_3_p, d01_3_p, d10_4_p, d01_4_p = (
        calculate_ZZ_XX_disp_pymablock(
            L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, pert_order=4
        )
    )

    print("\n================ COMPARISON RESULTS ================")
    print(f"Qubit 1 freq fq1 (GHz): Exact={fq1_e:.4f}, Pymablock={fq1_p:.4f}")
    print(f"Qubit 2 freq fq2 (GHz): Exact={fq2_e:.4f}, Pymablock={fq2_p:.4f}")
    print(f"Resonator 1 freq fc1 (GHz): Exact={fc1_e:.4f}, Pymablock={fc1_p:.4f}")
    print(f"Resonator 2 freq fc2 (GHz): Exact={fc2_e:.4f}, Pymablock={fc2_p:.4f}")

    print(f"ZZ Coupling (MHz): Exact={ZZ_e * 1000:.4f}, Pymablock={ZZ_p * 1000:.4f}")
    print(
        f"XX Exchange (MHz): BlackBox J12={XX_e * 1000:.4f}, Pymablock H_eff[1,2]={XX_p * 1000:.4f}"
    )

    print(
        f"disp_10_3 (MHz): Exact={d10_3_e * 1000:.4f}, Pymablock={d10_3_p * 1000:.4f}"
    )
    print(
        f"disp_01_3 (MHz): Exact={d01_3_e * 1000:.4f}, Pymablock={d01_3_p * 1000:.4f}"
    )
    print(
        f"disp_10_4 (MHz): Exact={d10_4_e * 1000:.4f}, Pymablock={d10_4_p * 1000:.4f}"
    )
    print(
        f"disp_01_4 (MHz): Exact={d01_4_e * 1000:.4f}, Pymablock={d01_4_p * 1000:.4f}"
    )

    # Verification assertions (within expected perturbation theory agreement)
    np.testing.assert_allclose(fq1_p, fq1_e, rtol=5e-2)
    np.testing.assert_allclose(fq2_p, fq2_e, rtol=5e-2)

    print("===================================================\n")


if __name__ == "__main__":
    test_compare_pymablock_and_exact_diagonalization()
