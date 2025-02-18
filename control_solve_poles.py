
from utils import PoleEquationsSolver
from utils import ChewMadelstemZero, KMatraixSumOfPoles, KMatraixSumOfPolesWOCoupling
import numpy as np


def main():
    at_inv = 7.219


    fac = 16 * np.pi # a factor due to the definition of t matrix basis

    p = {
        "g1": 526.2978085204057**.5 / fac**.5 / at_inv,
        "g2": 526.2978085204057**.5 / fac**.5 / at_inv,
        "M^2": 3.9797546469093215**2 / at_inv**2,
        "gamma11": -678.2656818319812 / fac,
        "gamma12": 0,
        "gamma22": -678.2656818319812 / fac,
    }

    m1_a = 1.896369299169502 / at_inv  # (0,1) are channels, (a,b) are particles
    m1_b = 2.017046814077588 / at_inv
    m2_a = 1.896369299169502 / at_inv  # (0,1) are channels, (a,b) are particles
    m2_b = 2.017046814077588 / at_inv

    pole_solver = PoleEquationsSolver()
    # Here KMatraixSumOfPolesWOCoupling is a test class for coupled channel without coupling.
    k_matrix_parameterization = KMatraixSumOfPolesWOCoupling(ChewMadelstemZero())
    k_matrix_parameterization.set_parameters(p)
    pole_solver.set_scattering_matrix(k_matrix_parameterization)

    print(pole_solver.get_pole_positions())
    # poles = pole_solver.get_pole_positions()
    pole_s0 = np.array([15.300732702657392]) / at_inv **2
    print("pole is ", pole_s0)

    # print K matrix at pole for checking
    K_matrix = pole_solver.scattering_matrix.get_K_matrix(pole_s0[0])
    print("K matrix at pole is\n", K_matrix * fac)

    # residues is a 1D array, index is channel1.
    residues_c0_square = pole_solver.get_t_matrix_pole_residues(pole_s0[0], m1_a, m1_b, m2_a, m2_b)
    print("residues c_0^2 * 16pi in GeV^2 is ", residues_c0_square * at_inv**2 * fac)  # 48.15163636461798
    print("END")

if __name__ == "__main__":
    main()