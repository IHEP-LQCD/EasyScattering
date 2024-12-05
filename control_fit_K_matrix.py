from utils import ScatteringDoubleChannelCalculator
from utils import ChewMadelstemZero, KMatraixSumOfPoles
import numpy as np

# MEMO
# Fit of a K matrix parameterization contains the two parts:
# 1. get chi2:
#   input: 1. a set of parameters to the K matrix parameterization,
#          2. a set of Jackknife data points
#   output: expected energy levels from the K matrix parameterization,
#           chi2 between the expected energy levels and the Jackknife data points.
# 2. iterate the parameters to minimize the chi2.


def main():
    calculator = ScatteringDoubleChannelCalculator(Ls=16, Q=np.ones(3) * 0.0, cut=30, xi_0=5.0)
    at_inv = 7.219

    # calculator.plot_zeta_function(calculator.M0000)

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
    energies_lat_data = np.load("./tests/jack_energy.npy").transpose((1, 0)) / at_inv
    calculator.set_resampling_energies(energies_lat_data**2, resampling_type="jackknife")
    print("Start fit chi2")
    chi2 = calculator.get_chi2(m1_A=m1_a, m1_B=m1_b, m2_A=m2_a, m2_B=m2_b)
    print(chi2)
    exit()


if __name__ == "__main__":
    main()
