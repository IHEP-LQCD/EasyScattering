from utils import DoubleChannelCalculator
from utils import ChewMadelstemZero, KMatraixSumOfPoles, KinvMatraixPolymomialSqrts
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
    calculator = DoubleChannelCalculator(Ls=16, Q=np.ones(3) * 0.0, cut=30, at_inv_GeV=7.219)
    at_inv = 7.219
    n_cfg = 401
    # calculator.plot_zeta_function(calculator.M0000)

    p_keys = ["c0_11", "c0_12", "c0_22", "c1_11", "c1_12", "c1_22", "c2_11", "c2_12", "c2_22"]
    p_values = [0] * 9
    p = dict(zip(p_keys, p_values))
    print(p)

    energies_lat_data = np.load("./tests/jack_energy_two_patricle.npy").transpose((1, 0))[:] / at_inv

    # this covariance matrix is used for jackknife resampling.
    # and it also internally used for chi2 calculation. show it here.
    cov = np.cov(energies_lat_data) * (n_cfg - 1) ** 2 / n_cfg  # convert numpy to jackknife cov

    # set K matrix parameterization
    k_matrix_parameterization = KinvMatraixPolymomialSqrts(ChewMadelstemZero())
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
    chi2 = calculator.get_chi2(p, verbose=True)
    print("time:", (perf_counter_ns() - s) / 1e9)
    print("chi2 = ", chi2)

    # iterate to minimize chi2
    from scipy.optimize import minimize

    def objective_function(params):
        param_dict = dict(zip(p_keys, params))
        return calculator.get_chi2(param_dict, verbose=False)

    para = list(p.values())
    best_chi2 = chi2
    best_para = para
    while chi2 > 10:
        para = [np.random.uniform(-0.005, 0.005) + i for i in para]
        result = minimize(objective_function, para, method="Nelder-Mead")
        print("Optimization result:", result)
        print("Minimized chi2:", result.fun)
        print("Optimized parameters:", result.x)
        if result.fun < best_chi2:
            best_chi2 = result.fun
            best_para = result.x
        print("Best chi2:", best_chi2)
        print("Best parameters:", ", ".join([str(i) for i in best_para]))

        chi2 = result.fun
        para = result.x


if __name__ == "__main__":
    main()
