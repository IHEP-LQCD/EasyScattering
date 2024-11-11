from utils import ScatteringDoubleChannelCalculator
from utils import ChewMadelstemZero, KMatraixSumOfPoles
import numpy as np

def main():
    calculator = ScatteringDoubleChannelCalculator(Ls=16, Q=np.ones(3) * 0.0, cut=30)

    # calculator.plot_zeta_function(calculator.M0000)

    chew_madelstem = ChewMadelstemZero()
    p = {
        "g1": 0.1,
        "g2": 0.2,
        "M^2": 3.91**2,
        "gamma11": 0.1,
        "gamma22": 0.7,
        "gamma12": 0.1,
        "gamma_p_11": 0.0,
        "gamma_p_22": 0.0,
        "gamma_p_12": 0.0,
        "m1A": 1.885,
        "m1B": 2.020,
        "m2A": 3.099,
        "m2B": 0.348,
    }
    m1_A = 1.885
    m1_B = 2.020
    m2_A = 3.099
    m2_B = 0.348
    k_matrix_parameterization = KMatraixSumOfPoles(p, chew_madelstem)
    calculator.set_scattering_matrix(k_matrix_parameterization)

    s = np.array([3.9, 4.0]) **2
    determinant = calculator.get_luescher_determint(s, m1_A=m1_A, m1_B=m1_B, m2_A=m2_A, m2_B=m2_B)
    print(determinant)
    calculator.scattering_matrix.get_S_matrix(s, m1_A=m1_A, m1_B=m1_B, m2_A=m2_A, m2_B=m2_B)
    exit()

if __name__ == "__main__":
    main()
