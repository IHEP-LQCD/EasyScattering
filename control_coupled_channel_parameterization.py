from utils import ScatteringDoubleChannelCalculator
from utils import ChewMadelstemZero, KMatraixSumOfPoles
import numpy as np


def main():
    calculator = ScatteringDoubleChannelCalculator(Ls=16, Q=np.ones(3) * 0.0, cut=30)
    at_inv = 6.894

    # calculator.plot_zeta_function(calculator.M0000)

    p = {
        "g1": 0.1,
        "g2": 0.2,
        "M^2": 3.91**2 / at_inv**2,
        "gamma11": 0.1,
        "gamma22": 0.7,
        "gamma12": 0.1,
        "gamma_p_11": 0.0,
        "gamma_p_22": 0.0,
        "gamma_p_12": 0.0,
    }
    m1_A = 1.885 / at_inv
    m1_B = 2.020 / at_inv
    m2_A = 3.099 / at_inv
    m2_B = 0.348 / at_inv
    k_matrix_parameterization = KMatraixSumOfPoles(ChewMadelstemZero())
    k_matrix_parameterization.set_parameters(p)
    calculator.set_scattering_matrix(k_matrix_parameterization)

    sqrt_s_GeV = np.linspace(3.9, 4.0, 100)
    s = (sqrt_s_GeV / at_inv) ** 2
    determinant = calculator.get_quantization_determinant(s, m1_A=m1_A, m1_B=m1_B, m2_A=m2_A, m2_B=m2_B)
    print(determinant)
    # calculator.scattering_matrix.get_S_matrix_from_t(s, m1_A=m1_A, m1_B=m1_B, m2_A=m2_A, m2_B=m2_B)
    exit()


if __name__ == "__main__":
    main()
