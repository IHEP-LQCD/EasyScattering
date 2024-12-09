import numpy as np
from numpy import ndarray
import gvar as gv
import cmath
from math import fsum
from functools import partial
from scipy.interpolate import interp1d
from typing import Union, List
import warnings
import os

from time import perf_counter

from .base import Analyticity, ChewMadelstemForm, ScatteringMatrixForm


class ScatteringDoubleChannelCalculator(Analyticity):
    """
    A simple methods class to calculate scattering amplitude.
        - Only for double Channel.
        - support fort both center-of-mass / rest frame and laboratory / moving frame.
        - design to support for ( L>=0 ) partial-wave (S,P,D...), but current only support for S-wave.
    Reference: 1) https://link.aps.org/doi/10.1103/PhysRevD.85.114507
               2) 10.1103/PhysRevD.88.014501
               3) some ChewMandelstam variable from above refs, and some from Journal of Mathematical Physics, (1998), 3540, 25(12)

    variables:
        L: lattice spacing, int
        Q: total momentum in lab frame, List[int]
        gamma: boost factor, float
        cut: Riemann Zeta function sumation cut off, sum over range [-cut: cut].
            int

    """

    Ls = None
    Q = None
    p = None

    def __init__(
        self,
        Ls: int,
        Q=[0, 0, 0],
        gamma: float = 1.0,
        cut: int = 30,
        xi_0=None,
        cache_file_dir: str = "./cache/",
    ) -> None:
        self.Ls = Ls
        self.Q = Q
        if list(Q) != [0, 0, 0]:
            raise NotImplementedError(f"Q = {Q} is not supported yet.")
        # gamma = 1.0 for rest frame

        self.gamma = gamma
        if xi_0 is None:
            xi_0 = 1.0
            warnings.warn("aspect_ratio xi_0 is not set, use default 1.0.", UserWarning)
        self.xi_0 = xi_0

        self.cut = cut if cut >= 30 else 30
        self.q2_begin = -9.0001
        self.q2_end = 11.0001
        self.q2_density = 2**18  # 262144
        self.cache_file_dir = cache_file_dir
        if not os.path.exists(cache_file_dir):
            os.makedirs(cache_file_dir)
        # self.fcn_M = self.factory_matrix_M(0, 0, 0, 0)

        self.scattering_matrix = None
        self.resampling_energies = None
        self.resampling_type = None
        self.n_resampling = None

        self.init_kM0000_cache()

    def _handle_resampling(self, p: Union[float, ndarray]):
        if isinstance(p, ndarray):
            if self.n_resampling is None:
                self.n_resampling = p.shape[0]
            elif self.n_resampling != p.shape[0]:
                raise ValueError(f"p.shape {p.shape} is not equal to n_resampling {self.n_resampling}.")
            return p
        return p * np.ones((self.n_resampling), "f8")

    def set_resampling_input(
        self,
        m1_A_resampling: Union[float, ndarray],
        m1_B_resampling: Union[float, ndarray],
        m2_A_resampling: Union[float, ndarray],
        m2_B_resampling: Union[float, ndarray],
        xi_0: Union[float, ndarray],
        n_resampling: int = None,
        resampling_type="jackknife",
    ):
        if self.n_resampling is None:
            self.n_resampling = n_resampling
        elif self.n_resampling != n_resampling:
            raise ValueError("n_resampling is set inconsistantly.")
        if resampling_type not in ["jackknife", "bootstrap"]:
            raise ValueError(f"resampling_type = {resampling_type} is not supported.")
        self.resampling_type = resampling_type
        self.m1_A_resampling = self._handle_resampling(m1_A_resampling)
        self.m1_B_resampling = self._handle_resampling(m1_B_resampling)
        self.m2_A_resampling = self._handle_resampling(m2_A_resampling)
        self.m2_B_resampling = self._handle_resampling(m2_B_resampling)
        self.m1_A_mean = np.mean(self.m1_A_resampling)
        self.m1_B_mean = np.mean(self.m1_B_resampling)
        self.m2_A_mean = np.mean(self.m2_A_resampling)
        self.m2_B_mean = np.mean(self.m2_B_resampling)
        self.xi_0_resampling = self._handle_resampling(xi_0)
        self.xi_0_mean = np.mean(self.xi_0_resampling)

    def init_kM0000_cache(self):
        cache_file_name = (
            f"{self.cache_file_dir}/cache_{self.q2_begin}_{self.q2_end}_{self.q2_density}_{self.cut}_L{self.Ls}.npy"
        )
        s = perf_counter()
        self.zeta_x = np.linspace(self.q2_begin, self.q2_end, self.q2_density, dtype="f8")
        if os.path.exists(cache_file_name):
            self.zeta_y = np.load(cache_file_name)
        else:
            print("Start init cache for zeta function, please wait.")
            # fcn_M = self.factory_matrix_M(0, 0, 0, 0)
            fcn_M = self.__kM0000_simpified
            self.zeta_y = fcn_M(self.zeta_x)
            np.save(cache_file_name, self.zeta_y)
        self.kM0000_interpolator = interp1d(self.zeta_x, self.zeta_y, kind="linear")
        print(f"INIT CACHE TIME: {perf_counter()-s:.3f} secs, file size = {self.zeta_y.nbytes/1024:.3f} KB")

    def __kM0000_simpified(self, q2):
        """
        simplified zeta function for S-wave.
        Note: M_0000 (k) = 2 / (sqrt(pi) * k * L) Z_00(1, q^2)
             L is dimension -1.
        Here:
            k * M_0000 = 2 / (sqrt(pi) * L) * Z_00(1, q^2)
                       = 1 / (pi * L) \sum_{n in R} [1 / (n^2 - q2)] - 4 pi R
        Note L = L_hat * a_s = 24 * a_s
            1/L = 1/24 * (at_inv_GeV / aspect_ratio)
        Here we use lattive unit that at_inv_GeV = 1 and energy levels = E * a_t.
        """
        cut = self.cut
        ret = 0
        for i in range(-cut, cut):
            for j in range(-cut, cut):
                for k in range(-cut, cut):
                    tmp = i**2 + j**2 + k**2
                    if tmp < cut**2:
                        ret += 1 / (tmp - q2)
        ret -= 4 * np.pi * cut
        return ret / (self.Ls * np.pi * self.xi_0)

    def set_scattering_matrix(self, form: ScatteringMatrixForm):
        """
        setter of scattering matrix form.
        """
        self.scattering_matrix = form

    # update: chew mandelstam form should be set in ScatteringMatrixForm.
    # def set_chew_mandelstam(self, form: ChewMadelstemForm = None):
    #     """
    #     settter of chew mandelstam form.
    #     """
    #     self.chew_mandelstam = form

    def set_resampling_energies(self, data: ndarray, resampling_type: str = "jackknife"):
        """
        setter of energy levels in resampling (jackknife / bootstrap) data.
            data: numpy.ndarray, shape = (n_levels, n_resampling)
        """
        if resampling_type not in ["jackknife", "bootstrap"]:
            raise ValueError(f"resampling_type = {resampling_type} is not supported.")
        self.resampling_type = resampling_type
        self.n_levels, self.n_resampling = data.shape
        # sort by mean value of each energy level.
        sorted_data = data[np.argsort(data.mean(axis=1))]
        self.resampling_energies = sorted_data
        print(f"init resampling_energies: n_levels = {self.n_levels}, n_resampling = {self.n_resampling}.")
        print("energy mean: ", sorted_data.mean(axis=1))

        # save the covariance matrix of resampling data.
        # dataset = gv.dataset.avg_data(energies_lat.transpose((1, 0)))
        # cov = gv.evalcov(dataset)
        cov = np.cov(self.resampling_energies)

        resampling_factor = self.resampling_type
        if resampling_factor == "jackknife":
            resampling_factor = (self.n_resampling - 1) ** 2 / self.n_resampling
        elif resampling_factor == "bootstrap":
            resampling_factor = 1
        else:
            raise ValueError(f"resampling_type = {resampling_factor} is not supported.")
        cov *= resampling_factor
        self.cov = cov
        self.cov_inv = np.linalg.inv(cov)

    def get_quantization_determint(self, s, m1_A, m1_B, m2_A, m2_B):
        """
        Det [K^-1 - diag(rho1 M0000, rho2 M0000)] = 0.
        rho M0000 = 2 k / sqrt(s) M0000 = 2/sqrt(s) * kM0000
        """
        K_inv = self.scattering_matrix.get_K_inv_matrix(s)

        # dimensionless: q2 = (k * L / 2 / np.pi) ** 2
        # a_s = aspect_ratio * a_t = aspect_ratio / at_inv_GeV
        q_square_1 = self.scattering_mom2(s, m1_A, m1_B) * (self.xi_0 * self.Ls / 2 / np.pi) ** 2
        q_square_2 = self.scattering_mom2(s, m2_A, m2_B) * (self.xi_0 * self.Ls / 2 / np.pi) ** 2
        rho_M0000_1 = 2 / np.vectorize(cmath.sqrt)(s) * self.kM0000_interpolator(q_square_1)
        rho_M0000_2 = 2 / np.vectorize(cmath.sqrt)(s) * self.kM0000_interpolator(q_square_2)
        if isinstance(s, ndarray):
            rho_M0000_matrix = np.zeros((s.shape[0], 2, 2), dtype="c16")
            rho_M0000_matrix[:, 0, 0] = rho_M0000_1
            rho_M0000_matrix[:, 1, 1] = rho_M0000_2
        else:
            rho_M0000_matrix = np.zeros((2, 2), dtype="f8")
            rho_M0000_matrix[0, 0] = rho_M0000_1
            rho_M0000_matrix[1, 1] = rho_M0000_2
        return np.linalg.det(K_inv - rho_M0000_matrix)

    def __quantization_determint_scale__(self, s, m1_A, m1_B, m2_A, m2_B) -> float:
        """
        Det [K^-1 - diag(rho1 M0000, rho2 M0000)] = 0.
        rho M0000 = 2 k / sqrt(s) M0000 = 2/sqrt(s) * kM0000
        """
        K_inv = self.scattering_matrix.get_K_inv_matrix(s)

        # dimensionless: q2 = (k * L / 2 / np.pi) ** 2
        # a_s = aspect_ratio * a_t = aspect_ratio / at_inv_GeV
        q_square_1 = self.scattering_mom2(s, m1_A, m1_B) * (self.xi_0 * self.Ls / 2 / np.pi) ** 2
        q_square_2 = self.scattering_mom2(s, m2_A, m2_B) * (self.xi_0 * self.Ls / 2 / np.pi) ** 2
        rho_M0000_1 = 2 / np.sqrt(s) * self.kM0000_interpolator(q_square_1)
        rho_M0000_2 = 2 / np.sqrt(s) * self.kM0000_interpolator(q_square_2)
        rho_M0000_matrix = np.zeros((2, 2), dtype="f8")
        # print(rho_M0000_1, rho_M0000_2)
        rho_M0000_matrix[0, 0] = rho_M0000_1
        rho_M0000_matrix[1, 1] = rho_M0000_2
        return np.linalg.det(K_inv - rho_M0000_matrix)

    @staticmethod
    def plot_zeros_search(x, y, zeros):
        import matplotlib.pyplot as plt

        plt.plot(x, y, "-b")
        for zero in zeros:
            plt.axvline(x=zero, color="r", linestyle="--")
        plt.plot(x, np.zeros_like(x), "-k")
        plt.ylim(-0.5, 0.5)
        plt.show()
        plt.clf()

    def get_quantization_determint_zeros(self, m1_A, m1_B, m2_A, m2_B, visiable=False):
        s0_zeros_prior = np.mean(self.resampling_energies, axis=1) ** 2

        n_levels = self.n_levels
        # s_zeros = np.zeros(n_levels)

        solve_upper = s0_zeros_prior[0] - 0.01
        solve_lower = s0_zeros_prior[-1] + 0.01
        n_point = 2**14
        interval = (solve_upper - solve_lower) / n_point

        s = np.linspace(solve_lower, solve_upper, n_point)
        quantization_determint = self.get_quantization_determint(s, m1_A, m1_B, m2_A, m2_B)
        zeros_index = quantization_determint**2 < 1e-6
        zeros_condidate = s[zeros_index]
        det_condidate = quantization_determint[zeros_index]
        s_zeros = []
        for idx in range(len(zeros_condidate)):
            if (
                (zeros_condidate[idx] - zeros_condidate[(idx + 1) % len(zeros_condidate)]) ** 2 < 25 * interval**2
            ) and (det_condidate[idx] * det_condidate[(idx + 1) % len(zeros_condidate)] < 0):
                s_zeros.append(zeros_condidate[idx])
        s_zeros = np.array(s_zeros)
        print(f"Find {s_zeros.shape} zeros: ", s_zeros)

        if visiable:
            self.plot_zeros_search(s**0.5 * 7.219, quantization_determint, s_zeros**0.5 * 7.219)


        # from scipy.optimize import fsolve
        # fcn = partial(self.__quantization_determint_scale__, m1_A=m1_A, m1_B=m1_B, m2_A=m2_A, m2_B=m2_B)

        # for i in range(n_levels):
        #     try:
        #         zero = fsolve(fcn, s0_zeros_prior[i])
        #     except Exception as e:
        #         print(f"Error in fsolve for level {i}: {e}")
        #         zero = [0]
        #     s_zeros[i] = zero[0]
        print("E_exp: \t", s_zeros**0.5 * 7.219)
        print("E_lat: \t", s0_zeros_prior**0.5 * 7.219)
        return s_zeros**0.5

    def get_chi2(self, p=None, cov=None):
        """
        get chi2 between the expected energy levels from the K matrix parameterization and the Jackknife data points.
        """
        if p is not None:
            self.scattering_matrix.set_parameters(p)
        if self.scattering_matrix._p is None:
            raise ValueError("ScatteringMatrix parameters is not set yet, set_parameters(p) first.")
        if self.resampling_energies is None:
            raise ValueError("resampling_energies is not set yet.")
        m1_A, m1_B, m2_A, m2_B = self.m1_A_mean, self.m1_B_mean, self.m2_A_mean, self.m2_B_mean
        energies_lat = self.resampling_energies
        n_levels = self.n_levels

        energies_exp = self.get_quantization_determint_zeros(m1_A, m1_B, m2_A, m2_B)
        print(energies_exp * 7.219)

        cov_inv = self.cov_inv
        if cov is not None:
            cov_inv = np.linalg.inv(cov)

        energies_lat_mean = np.mean(energies_lat, axis=1)
        chi2 = np.einsum("i, ij, j", energies_exp - energies_lat_mean, cov_inv, energies_exp - energies_lat_mean)
        # print("return chi2 = ", chi2)
        return chi2

    @staticmethod
    def plot_zeta_function(fcn):
        """
        plot to check behavior.
        """
        x = np.linspace(-0.9, 8.5, 10000)
        y = fcn(x)
        import matplotlib.pyplot as plt

        plt.plot(x, y, "--b")
        plt.ylim(-10, 10)
        plt.show()
        plt.clf()

    def plot_luescher_determint(self, s, m1_A, m1_B, m2_A, m2_B, x):
        """
        plot to check behavior.
        """
        determinant = self.get_quantization_determint(s, m1_A, m1_B, m2_A, m2_B)
        import matplotlib.pyplot as plt

        plt.plot(x, determinant, "-b")
        plt.ylim(-1, 1)
        plt.show()
        plt.clf()
