from utils import ScatteringDoubleChannelCalculator
from utils import ChewMadelstemZero, KMatraixSumOfPoles
import numpy as np
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
    calculator = ScatteringDoubleChannelCalculator(Ls=16, Q=np.ones(3) * 0.0, cut=30, xi_0=5.0)
    at_inv = 7.219
    n_cfg = 401

    # calculator.plot_zeta_function(calculator.M0000)

    p = {
        "g1": 1.5467876391135578 / at_inv,
        "g2": -3.0927332462193746 / at_inv,
        "M^2": 42.03222752610351 / at_inv**2,
        "gamma11": 1.6874087208657231,
        "gamma22": -2.8785928892611397,
        "gamma12": 4.882070870322092,
    }

    k_matrix_parameterization = KMatraixSumOfPoles(ChewMadelstemZero())
    k_matrix_parameterization.set_parameters(p)
    calculator.set_scattering_matrix(k_matrix_parameterization)
    energies_lat_data = np.load("./tests/jack_energy_two_patricle.npy").transpose((1, 0)) / at_inv
    calculator.set_resampling_energies(energies_lat_data, resampling_type="jackknife")

    m1_resampling = np.load("./tests/jack_energy_single.npy")[:, 0] / at_inv
    m2_resampling = np.load("./tests/jack_energy_single.npy")[:, 5] / at_inv
    calculator.set_resampling_input(
        m1_A_resampling=m1_resampling,
        m1_B_resampling=m1_resampling,
        m2_A_resampling=m2_resampling,
        m2_B_resampling=m2_resampling,
        n_resampling=n_cfg,
        xi_0=5.0,
        resampling_type="jackknife",
    )
    print("Start fit chi2")
    s = perf_counter_ns()
    chi2 = calculator.get_chi2(p)
    print("time:", (perf_counter_ns() - s) / 1e9)
    print("chi2 = ", chi2)
    # exit()

    from scipy.optimize import minimize

    def objective_function(params):
        param_dict = {
            "g1": params[0],
            "g2": params[1],
            "M^2": params[2],
            "gamma11": params[3],
            "gamma22": params[4],
            "gamma12": params[5],
        }
        # print("input p = \n", param_dict)
        return calculator.get_chi2(param_dict)

    para = [p["g1"], p["g2"], p["M^2"], p["gamma11"], p["gamma22"], p["gamma12"]]
    # print(para)
    result = minimize(objective_function, para)
    print("Optimization result:", result)
    print("Minimized chi2:", result.fun)
    print("Optimized parameters:", result.x)
    exit()


if __name__ == "__main__":
    main()
