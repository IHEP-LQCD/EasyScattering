from utils import DoubleChannelCalculator, JointDoubleChannelCalculator
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
    calculator_L16 = DoubleChannelCalculator(Ls=16, Q=np.ones(3) * 0.0, cut=30, at_inv_GeV=7.219)
    calculator_L24 = DoubleChannelCalculator(Ls=24, Q=np.ones(3) * 0.0, cut=30, at_inv_GeV=7.219)
    at_inv = 7.219
    n_cfg = 36

    p_keys = ["g1", "g2", "M^2", "gamma11", "gamma22", "gamma12"]
    p_values = [0.1] * 6
    p = dict(zip(p_keys, p_values))

    energies_lat_data_L16 = np.load("") / at_inv
    energies_lat_data_L24 = np.load("") / at_inv
    assert energies_lat_data_L16.shape == (10, n_cfg)
    assert energies_lat_data_L24.shape == (10, n_cfg)

    # cov = np.cov(energies_lat_data_L16) * (n_cfg - 1) ** 2 / n_cfg  # convert numpy to jackknife cov
    # cov = np.diag(np.diag(cov))

    k_matrix_parameterization = KMatraixSumOfPoles(ChewMadelstemZero())
    k_matrix_parameterization.set_parameters(p)
    calculator_L16.set_scattering_matrix(k_matrix_parameterization)
    calculator_L16.set_resampling_energies(energies_lat_data_L16, resampling_type="jackknife")
    calculator_L24.set_scattering_matrix(k_matrix_parameterization)
    calculator_L24.set_resampling_energies(energies_lat_data_L24, resampling_type="jackknife")

    m1_resampling_l16 = np.load("./tests/jack_energy_L16.npy")[:, 0] / at_inv  # eta_c
    m2_resampling_L16 = np.load("./tests/jack_energy_L16.npy")[:, 5] / at_inv  # jpsi
    calculator_L16.set_resampling_input(
        m1_A_resampling=m1_resampling_l16,
        m1_B_resampling=m1_resampling_l16,
        m2_A_resampling=m2_resampling_L16,
        m2_B_resampling=m2_resampling_L16,
        n_resampling=n_cfg,
        # xi=5.0,
        xi1=5.0,
        xi2=5.0,
        resampling_type="jackknife",
    )

    m1_resampling_l24 = np.load("./tests/jack_energy_2_L24.npy")[:, 0] / at_inv  # eta_c
    m2_resampling_L24 = np.load("./tests/jack_energy_2_L24.npy")[:, 5] / at_inv  # jpsi
    calculator_L24.set_resampling_input(
        m1_A_resampling=m1_resampling_l24,
        m1_B_resampling=m1_resampling_l24,
        m2_A_resampling=m2_resampling_L24,
        m2_B_resampling=m2_resampling_L24,
        n_resampling=n_cfg,
        # xi=5.0,
        xi1=5.0,
        xi2=5.0,
        resampling_type="jackknife",
    )

    calculator_joint = JointDoubleChannelCalculator([calculator_L16, calculator_L24])

    print("Start fit chi2")
    s = perf_counter_ns()
    chi2 = calculator_joint.get_chi2(p, verbose=True)
    print("time:", (perf_counter_ns() - s) / 1e9)
    print("chi2 = ", chi2)

    # iterate to minimize chi2
    from scipy.optimize import minimize

    def objective_function(params):
        param_dict = dict(zip(p_keys, params))
        return calculator_joint.get_chi2(param_dict, verbose=False)

    print(chi2)
    para = list(p.values())
    best_chi2 = chi2
    best_para = para
    while chi2 > 10:
        para = [np.random.uniform(-0.1, 0.1) + i for i in para]
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
    exit()


if __name__ == "__main__":
    main()
