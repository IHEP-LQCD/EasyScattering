from utils import ScatteringDoubleChannelCalculator
from utils import ChewMadelstemZero, KMatraixSumOfPoles
import numpy as np
import gvar as gv
from time import perf_counter_ns

# MEMO
# Fit of a K matrix parameterization contains the two parts:
# 1. get chi2:
#   input: 1. a set of parameters to the K matrix parameterization,
#          2. a set of Jackknife data points
#   output: expected energy levels from the K matrix parameterization,
#           chi2 between the expected energy levels and the Jackknife data points.
# 2. iterate the parameters to minimize the chi2.


def main():
    calculator = ScatteringDoubleChannelCalculator(Ls=16, Q=np.ones(3) * 0.0, cut=30, at_inv_GeV=7.219)
    at_inv = 7.219
    n_cfg = 401

    # calculator.plot_zeta_function(calculator.M0000)

    # p_keys order is consistent with the order of p_values
    p_keys = ["g1", "g2", "M^2", "gamma11", "gamma22", "gamma12"]
    # p_values is a test value, you can set any init value as prior.
    p_values = [0.87375412, -1.0066831, 1.7326207, -1.20996452, -1.25834206, 2.77249789]
    p = dict(zip(p_keys, p_values))

    energies_lat_data = np.load("./tests/jack_energy_two_patricle.npy").transpose((1, 0)) / at_inv

    # this covariance matrix is used for jackknife resampling.
    # and it also internally used for chi2 calculation. show it here.
    cov = np.cov(energies_lat_data) * (n_cfg - 1) ** 2 / n_cfg  # convert numpy to jackknife cov

    k_matrix_parameterization = KMatraixSumOfPoles(ChewMadelstemZero())
    k_matrix_parameterization.set_parameters(p)
    calculator.set_scattering_matrix(k_matrix_parameterization)
    calculator.set_resampling_energies(energies_lat_data, resampling_type="jackknife")

    m1_resampling = np.load("./tests/jack_energy_single.npy")[:, 0] / at_inv  # eta_c
    m2_resampling = np.load("./tests/jack_energy_single.npy")[:, 5] / at_inv  # jpsi
    calculator.set_resampling_input(
        m1_A_resampling=m1_resampling,
        m1_B_resampling=m1_resampling,
        m2_A_resampling=m2_resampling,
        m2_B_resampling=m2_resampling,
        n_resampling=n_cfg,
        # xi=5.0, # if set constant dispersion relation.
        xi1=np.load("./tests/M415_etac_xi_jack.npy"),  # set resampling dispersion relations.
        xi2=np.load("./tests/M415_jpsi_xi_jack.npy"),  # set resampling dispersion relations.
        resampling_type="jackknife",
    )
    print("Start fit chi2")
    s = perf_counter_ns()
    # set cov for hack covariance, use it for debugging, but not recommended for real fit.
    # if you want to see the E_expected and the quantization determinant zeros distribution, set verbose=True
    chi2 = calculator.get_chi2(p, verbose=True)
    print("time:", (perf_counter_ns() - s) / 1e9)
    print("chi2 = ", chi2)

    # resampling to chi2
    chi2_resampling, parametrs_resampling, energies_exp_resampling = calculator.get_chi2_resampling(p, verbose=True)
    print("chi2_resampling = ", chi2_resampling)

    exit()


if __name__ == "__main__":
    main()
