from utils import ScatteringDoubleChannelCalculator
from utils import ChewMadelstemZero, KMatraixSumOfPoles, KinvMatraixPolymomialSqrts
import numpy as np
import gvar as gv
from time import perf_counter_ns
from tqdm import trange

# MEMO
# K matrix parameters -> K / t / Kinv /tinv matrix -> Scattering matrix

p_keys = ["g1", "g2", "M^2", "gamma11", "gamma22", "gamma12"]
K_matrix_parameters = np.load("output/L16_chi2_parameters_resampling.npy")

def get_gvar(p):
    return gv.gvar(p.mean(0), p.std(0) * np.sqrt(p.shape[0] - 1))

for i in range(len(p_keys)):
    print(p_keys[i], get_gvar(K_matrix_parameters[:, i]))

at_inv = 7.219
n_resampling = 401

parameterization = KMatraixSumOfPoles(ChewMadelstemZero())

# for eaxmple, print the t matrix, tinverse matrix and S matrix.
sqrt_s_GEV = np.linspace(5.5, 7.0, 1024)
s = (sqrt_s_GEV / at_inv) ** 2


rhot_square_abs_jack = np.zeros((n_resampling, len(s), 2, 2))
for i in trange(n_resampling):
    parameterization.set_parameters(dict(zip(p_keys, K_matrix_parameters[i])))
    rhot_square_abs_jack[i] = parameterization.get_rhot_square_abs(
        s, m1_A=2.99528 / at_inv, m1_B=2.99528 / at_inv, m2_A=3.08699 / at_inv, m2_B=3.08699 / at_inv
    )

rhot_square_abs_mean = np.mean(rhot_square_abs_jack, axis=0)
rhot_square_abs_err = np.std(rhot_square_abs_jack, axis=0) * np.sqrt(n_resampling - 1)

import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

plt.plot(sqrt_s_GEV, rhot_square_abs_mean[:, 0, 0], "r", label=r"$\eta_c \eta_c \to \eta_c \eta_c$")
print(rhot_square_abs_mean[:, 0, 0])
print(rhot_square_abs_err[:, 0, 0])
plt.fill_between(
    sqrt_s_GEV,
    rhot_square_abs_mean[:, 0, 0] - rhot_square_abs_err[:, 0, 0],
    rhot_square_abs_mean[:, 0, 0] + rhot_square_abs_err[:, 0, 0],
    color="r",
    alpha=0.2,
)

plt.plot(sqrt_s_GEV, rhot_square_abs_mean[:, 1, 1], "g", label=r"$J/\psi J/\psi \to J/\psi J/\psi$")
plt.fill_between(
    sqrt_s_GEV,
    rhot_square_abs_mean[:, 1, 1] - rhot_square_abs_err[:, 1, 1],
    rhot_square_abs_mean[:, 1, 1] + rhot_square_abs_err[:, 1, 1],
    color="g",
    alpha=0.2,
)

plt.plot(sqrt_s_GEV, rhot_square_abs_mean[:, 0, 1], "b", label=r"$\eta_c \eta_c \to J/\psi J/\psi$")
plt.fill_between(
    sqrt_s_GEV,
    rhot_square_abs_mean[:, 0, 1] - rhot_square_abs_err[:, 0, 1],
    rhot_square_abs_mean[:, 0, 1] + rhot_square_abs_err[:, 0, 1],
    color="b",
    alpha=0.2,
)

plt.ylim(0, 0.6)
plt.xlabel(r"$\sqrt{s}$ [GeV]", fontsize=15)
plt.ylabel(r"$|\rho t|^2$", fontsize=15)
plt.legend(loc="upper left")
plt.show()
plt.clf()
