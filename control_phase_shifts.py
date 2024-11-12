from utils import ScatteringDoubleChannelCalculator
from utils import ChewMadelstemZero, KMatraixSumOfPoles
import numpy as np

def main():
    calculator = ScatteringDoubleChannelCalculator(Ls=16, Q=np.ones(3) * 0.0, cut=30)

    # calculator.plot_zeta_function(calculator.M0000)

    chew_madelstem = ChewMadelstemZero()
    # p = {
    #     "g1": 0.1,
    #     "g2": 0.2,
    #     "M^2": 3.91**2,
    #     "gamma11": 0.1,
    #     "gamma22": 0.7,
    #     "gamma12": 0.1,
    #     "gamma_p_11": 0.0,
    #     "gamma_p_22": 0.0,
    #     "gamma_p_12": 0.0,
    #     "m1A": 1.885,
    #     "m1B": 2.020,
    #     "m2A": 3.099,
    #     "m2B": 0.348,
    # }

    at_inv = 7.219
    p = {'g1': 1.5467876391135578  / at_inv,
         'g2': -3.0927332462193746 / at_inv,
         'M^2': 42.03222752610351 / at_inv**2,
         'gamma11': 1.6874087208657231,
         'gamma22': -2.8785928892611397,
         'gamma12': 4.882070870322092}

    # m1_A = 1.885
    # m1_B = 2.020
    # m2_A = 3.099
    # m2_B = 0.348
    m1_a = 2.99528 / at_inv  # (0,1) are channels, (a,b) are particles
    m1_b = 2.99528 / at_inv
    m2_a = 3.08699 / at_inv
    m2_b = 3.08699 / at_inv
    k_matrix_parameterization = KMatraixSumOfPoles(p, chew_madelstem)
    calculator.set_scattering_matrix(k_matrix_parameterization)

    # s = np.array([3.9, 4.0]) **2
    sqrt_s_GEV = np.linspace(5.5, 7.0, 10000)
    s = (sqrt_s_GEV / at_inv) ** 2
    # determinant = calculator.get_luescher_determint(s, m1_A=m1_A, m1_B=m1_B, m2_A=m2_A, m2_B=m2_B)
    # print(determinant)
    s_matrix = calculator.scattering_matrix.get_S_matrix(s, m1_A=m1_a, m1_B=m1_b, m2_A=m2_a, m2_B=m2_b)
    # exit()
    # phase = calculator.scattering_matrix.get_phase_from_S_matrix(s_matrix)
    calculator.scattering_matrix.plot_phase(s_matrix, sqrt_s_GEV)
    exit()

if __name__ == "__main__":
    main()
