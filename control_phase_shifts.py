from utils import DoubleChannelCalculator
from utils import ChewMadelstemZero, KMatraixSumOfPoles
import numpy as np


def main():
    at_inv = 7.219
    calculator = DoubleChannelCalculator(
        Ls=16, Q=np.ones(3) * 0.0, cut=30, at_inv_GeV=7.219
    )

    p = {
        "g1": 1.5467876391135578 / at_inv,
        "g2": -3.0927332462193746 / at_inv,
        "M^2": 42.03222752610351 / at_inv**2,
        "gamma11": 1.6874087208657231,
        "gamma22": -2.8785928892611397,
        "gamma12": 4.882070870322092,
    }
    m1_a = 2.99528 / at_inv  # (0,1) are channels, (a,b) are particles
    m1_b = 2.99528 / at_inv
    m2_a = 3.08699 / at_inv
    m2_b = 3.08699 / at_inv
    k_matrix_parameterization = KMatraixSumOfPoles(ChewMadelstemZero())
    k_matrix_parameterization.set_parameters(p)
    calculator.set_scattering_matrix(k_matrix_parameterization)

    # for eaxmple, print the t matrix, tinverse matrix and S matrix.
    sqrt_s_GEV = np.linspace(5.5, 7.0, 15)
    s = (sqrt_s_GEV / at_inv) ** 2
    t_inv = calculator.scattering_matrix.get_t_inv_matrix(
        s, m1_A=m1_a, m1_B=m1_b, m2_A=m2_a, m2_B=m2_b
    )

    # check the t_inv.imag == -rho
    assert np.allclose(
        t_inv.imag, -calculator.rho_matrix(s, m1_a, m1_b, m2_a, m2_b).real
    )

    # assert np.allclose( - 1j*(t_inv - t_inv.conj()), 2 * calculator.rho_matrix(s, m1_a, m1_b, m2_a, m2_b))
    print(
        1j * (t_inv - t_inv.conj()) + calculator.rho_matrix(s, m1_a, m1_b, m2_a, m2_b)
    )
    # exit()
    print("tinv = \n", t_inv)

    t = calculator.scattering_matrix.get_t_matrix(
        s, m1_A=m1_a, m1_B=m1_b, m2_A=m2_a, m2_B=m2_b
    )
    print("t = \n", t)

    S = calculator.scattering_matrix.get_S_matrix_from_t(
        s, m1_A=m1_a, m1_B=m1_b, m2_A=m2_a, m2_B=m2_b
    )
    print("S = \n", S)


    # plot the phase shift
    sqrt_s_GEV = np.linspace(5.5, 7.0, 10000)
    s = (sqrt_s_GEV / at_inv) ** 2


    determinant = calculator.get_quantization_determinant(s, m1_A=m1_a, m1_B=m1_b, m2_A=m2_a, m2_B=m2_b)
    calculator.plot_quantization_determinant(s, m1_a, m1_b, m2_a, m2_b, sqrt_s_GEV)
    exit()

    s_matrix = calculator.scattering_matrix.get_S_matrix_from_t(
        s, m1_A=m1_a, m1_B=m1_b, m2_A=m2_a, m2_B=m2_b
    )

    # s_matrix = calculator.scattering_matrix.get_S_matrix_from_K(s)

    # phase = calculator.scattering_matrix.get_phase_from_S_matrix(s_matrix)
    calculator.scattering_matrix.plot_phase(s_matrix, sqrt_s_GEV)

    calculator.scattering_matrix.plot_rhot_square(s, m1_a, m1_b, m2_a, m2_b, sqrt_s_GEV)
    exit()


if __name__ == "__main__":
    main()
