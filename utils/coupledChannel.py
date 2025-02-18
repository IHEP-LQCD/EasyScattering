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


from scipy.optimize import minimize
from tqdm import trange

from time import perf_counter

from .base import ScatteringMatrixABC, ScatteringCalculatorABC


class DoubleChannelCalculator(ScatteringCalculatorABC):
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
        at_inv_GeV: float = 1.0,
        cache_file_dir: str = "./cache/",
        name: str = "",
        search_n_points: int = 2**10,
    ) -> None:
        """
        init method.
            search_n_points: number of points to search zeros.
                almost proprotional to cost of time.
        """
        self.Ls = Ls
        self.Q = Q
        if list(Q) != [0, 0, 0]:
            raise NotImplementedError(f"Q = {Q} is not supported yet.")
        # gamma = 1.0 for rest frame

        self.gamma = gamma
        self.name = name

        self.cut = cut if cut >= 30 else 30
        self.q2_begin = -10
        self.q2_end = 11
        # self.q2_interp1d_density = 2**18  # 262144
        self.q2_interp1d_density = 2**14  # interp1d density in each q2 interval.
        self._solve_zeros_n_point = search_n_points
        self.cache_file_dir = cache_file_dir
        if not os.path.exists(cache_file_dir):
            os.makedirs(cache_file_dir)
        # self.fcn_M = self.factory_matrix_M(0, 0, 0, 0)
        self.scattering_matrix = None
        self.resampling_energies = None
        self.resampling_type = None
        self.n_resampling = None
        self.resampling_factor = None
        self.cov = None
        self.cov_inv = None

        # set at_inv_GeV only to display the energy in GeV and do nothing else.
        self.at_inv_GeV = at_inv_GeV

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
        xi: Union[float, ndarray] = None,
        xi1: Union[float, ndarray] = None,
        xi2: Union[float, ndarray] = None,
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
        if xi is not None and xi1 is None and xi2 is None:
            self.xi1_resampling = self._handle_resampling(xi)
            self.xi1_mean = np.mean(self.xi1_resampling)
            self.xi2_resampling = self._handle_resampling(xi)
            self.xi2_mean = np.mean(self.xi2_resampling)
        elif xi1 is not None:
            self.xi1_resampling = self._handle_resampling(xi1)
            self.xi1_mean = np.mean(self.xi1_resampling)
            self.xi2_resampling = self._handle_resampling(xi2)
            self.xi2_mean = np.mean(self.xi2_resampling)
        else:
            raise ValueError("xi should be set.")

        # solve zeros resolution
        _s0_zeros_prior = np.mean(self.resampling_energies, axis=1) ** 2
        _solve_zeros_lower = _s0_zeros_prior[0] - 0.01
        _solve_zeros_upper = _s0_zeros_prior[-1] + 0.01

        focus_on_list = [_solve_zeros_lower, _solve_zeros_upper]
        for i in range(self.n_levels):
            e1 = (self.m1_A_mean**2 + i * (2 * np.pi / self.Ls / self.xi1_mean) ** 2) ** 0.5 + (
                self.m1_B_mean**2 + i * (2 * np.pi / self.Ls / self.xi1_mean) ** 2
            ) ** 0.5
            e2 = (self.m2_A_mean**2 + i * (2 * np.pi / self.Ls / self.xi2_mean) ** 2) ** 0.5 + (
                self.m2_B_mean**2 + i * (2 * np.pi / self.Ls / self.xi2_mean) ** 2
            ) ** 0.5
            focus_on_list.append(e1**2)
            focus_on_list.append(e2**2)
        focus_on_list = np.sort(focus_on_list)
        print(f"focus on energy on {self.name}\n", np.sort([i**0.5 * self.at_inv_GeV for i in focus_on_list]))

        _n_point = self._solve_zeros_n_point
        # weight = 2**8 / (2**4)
        # self._solve_zeros_interval = (_solve_zeros_upper - _solve_zeros_lower) / _n_point
        # self._solve_zeros_s_linspace = np.linspace(_solve_zeros_lower, _solve_zeros_upper, _n_point)
        self._solve_zeros_s_linspace = np.concatenate(
            [np.linspace(focus_on_list[i], focus_on_list[i + 1], _n_point) for i in range(len(focus_on_list) - 1)]
        )

    def init_kM0000_cache(self):
        cache_file_name = f"{self.cache_file_dir}/cache_{self.q2_begin}_{self.q2_end}_{self.q2_interp1d_density}_{self.cut}_L{self.Ls}.npy"
        s = perf_counter()
        _density = self.q2_interp1d_density
        weight = _density / (2**4)
        interp1d_int = np.concatenate(
            [
                np.exp(-np.arange(_density, 0, -1) / weight) / 2,
                (2 - np.exp(-np.arange(1, _density + 1, 1) / weight)) / 2,
            ],
            dtype="f8",
        )
        self.zeta_x = np.concatenate([interp1d_int + i for i in range(self.q2_begin, self.q2_end)], dtype="f8")
        # self.zeta_x = np.linspace(self.q2_begin, self.q2_end, self.q2_interp1d_density, dtype="f8")
        if os.path.exists(cache_file_name):
            self.zeta_y = np.load(cache_file_name)
        else:
            print("Start init cache for zeta function, please wait.")
            # fcn_M = self.factory_matrix_M(0, 0, 0, 0)
            fcn_M = self.__kM0000_times_xi_simpified
            self.zeta_y = fcn_M(self.zeta_x)
            np.save(cache_file_name, self.zeta_y)
        self.kM0000_times_xi_interpolator = interp1d(self.zeta_x, self.zeta_y, kind="linear")
        print(f"INIT CACHE TIME: {perf_counter()-s:.3f} secs, file size = {self.zeta_y.nbytes/1024:.3f} KB")

    def __kM0000_times_xi_simpified(self, q2):
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
        return ret / (self.Ls * np.pi)

    def set_scattering_matrix(self, form: ScatteringMatrixABC):
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
        self.resampling_factor = resampling_factor
        self.cov = cov
        self.cov_inv = np.linalg.inv(cov)
        print(f"---- {self.name} cov_inv ---- \n", self.cov_inv)

    def get_quantization_determinant(self, s, m1_A, m1_B, m2_A, m2_B, xi1, xi2):
        """
        Det [K^-1 - diag(rho1 M0000, rho2 M0000)] = 0.
        rho M0000 = 2 k / sqrt(s) M0000 = 2/sqrt(s) * kM0000
        """
        K_inv = self.scattering_matrix.get_K_inv_matrix(s)

        # dimensionless: q2 = (k * L / 2 / np.pi) ** 2
        # a_s = aspect_ratio * a_t = aspect_ratio / at_inv_GeV
        q_square_1 = self.scattering_mom2(s, m1_A, m1_B) * (xi1 * self.Ls / 2 / np.pi) ** 2
        q_square_2 = self.scattering_mom2(s, m2_A, m2_B) * (xi2 * self.Ls / 2 / np.pi) ** 2
        rho_M0000_1 = 2 / np.vectorize(cmath.sqrt)(s) * self.kM0000_times_xi_interpolator(q_square_1) / xi1
        rho_M0000_2 = 2 / np.vectorize(cmath.sqrt)(s) * self.kM0000_times_xi_interpolator(q_square_2) / xi1
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
        q_square_1 = self.scattering_mom2(s, m1_A, m1_B) * (self.xi1_mean * self.Ls / 2 / np.pi) ** 2
        q_square_2 = self.scattering_mom2(s, m2_A, m2_B) * (self.xi2_mean * self.Ls / 2 / np.pi) ** 2
        rho_M0000_1 = 2 / np.sqrt(s) * self.kM0000_times_xi_interpolator(q_square_1)
        rho_M0000_2 = 2 / np.sqrt(s) * self.kM0000_times_xi_interpolator(q_square_2)
        rho_M0000_matrix = np.zeros((2, 2), dtype="f8")
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
        plt.ylim(-5, 5)
        plt.show()
        plt.clf()

    def get_quantization_determinant_zeros(self, visiable=False):
        s = self._solve_zeros_s_linspace
        m1_A, m1_B, m2_A, m2_B = self.m1_A_mean, self.m1_B_mean, self.m2_A_mean, self.m2_B_mean

        start_time = perf_counter()
        quantization_determint = self.get_quantization_determinant(
            s, m1_A, m1_B, m2_A, m2_B, xi1=self.xi1_mean, xi2=self.xi2_mean
        )
        zeros_index = (quantization_determint**2 < 1e-3) & (
            quantization_determint * np.roll(quantization_determint, -1) < 0
        )
        s_zeros = s[zeros_index]
        if visiable:
            self.plot_zeros_search(s**0.5 * self.at_inv_GeV, quantization_determint, s_zeros**0.5 * self.at_inv_GeV)
            print(f"Find {s_zeros.shape} zeros: ", s_zeros)

        return s_zeros**0.5

    def get_quantization_determinant_zeros_resampling(self, visiable=False, i_resampling: int = 0):
        s = self._solve_zeros_s_linspace
        # m1_A, m1_B, m2_A, m2_B = self.m1_A_mean, self.m1_B_mean, self.m2_A_mean, self.m2_B_mean
        m1_A, m1_B, m2_A, m2_B = (
            self.m1_A_resampling[i_resampling],
            self.m1_B_resampling[i_resampling],
            self.m2_A_resampling[i_resampling],
            self.m2_B_resampling[i_resampling],
        )
        quantization_determint = self.get_quantization_determinant(
            s, m1_A, m1_B, m2_A, m2_B, xi1=self.xi1_resampling[i_resampling], xi2=self.xi2_resampling[i_resampling]
        )
        zeros_index = (quantization_determint**2 < 1e-3) & (
            quantization_determint * np.roll(quantization_determint, -1) < 0
        )
        s_zeros = s[zeros_index]

        if visiable:
            self.plot_zeros_search(s**0.5 * self.at_inv_GeV, quantization_determint, s_zeros**0.5 * self.at_inv_GeV)
            print(f"Find {s_zeros.shape} zeros: ", s_zeros)

        return s_zeros**0.5

    def get_chi2(self, p=None, cov_debugging=None, is_nearest_zeros=True, verbose=False):
        """
        get chi2 between the expected energy levels from the K matrix parameterization.
            cov_debugging: covariance matrix for debugging, not recommended for real fit.
        """
        if p is not None:
            self.scattering_matrix.set_parameters(p)
        if self.scattering_matrix._p is None:
            raise ValueError("ScatteringMatrix parameters is not set yet, set_parameters(p) first.")
        if self.resampling_energies is None:
            raise ValueError("resampling_energies is not set yet.")
        energies_lat = self.resampling_energies
        n_levels = self.n_levels

        energies_at_zeros = self.get_quantization_determinant_zeros(visiable=False)

        cov_inv = self.cov_inv
        if cov_debugging is not None:
            cov_inv = np.linalg.inv(cov_debugging)

        energies_lat_mean = np.mean(energies_lat, axis=1)
        energies_exp = np.zeros(n_levels)
        if is_nearest_zeros:
            if len(energies_at_zeros) == 0:
                energies_exp[:] = 0
            else:
                if len(energies_at_zeros) < n_levels:
                    self.get_quantization_determinant_zeros(visiable=True)
                    raise ValueError(f"zeros number {len(energies_at_zeros)} is less than n_levels {n_levels}.")
                for ie in range(n_levels):
                    idx = np.argmin(np.abs(energies_lat_mean[ie] - energies_at_zeros))
                    energies_exp[ie] = energies_at_zeros[idx]
                # for ie in range(len(energies_at_zeros)):
                #
        else:
            if len(energies_at_zeros) < n_levels:
                self.get_quantization_determinant_zeros(visiable=True)
                raise ValueError(f"zeros number {len(energies_at_zeros)} is less than n_levels {n_levels}.")
            energies_exp[:] = energies_at_zeros[:n_levels]
            print(energies_exp.shape, energies_lat_mean.shape, n_levels)

        chi2 = np.einsum("i, ij, j", energies_exp - energies_lat_mean, cov_inv, energies_exp - energies_lat_mean)
        if verbose:
            print(f"--------- {self.name} chi2 = {chi2:.4} ----------")
            print("E_exp:\t", "\t".join(f"{i:.9}" for i in energies_exp * self.at_inv_GeV))
            print(
                "E_lat:\t",
                "\t".join(
                    f"{i}"
                    for i in gv.gvar(energies_lat.mean(axis=1), energies_lat.std(axis=1) * self.resampling_factor**0.5)
                    * self.at_inv_GeV
                ),
            )
            tmp = np.einsum(
                "i, ij, j -> ij", energies_exp - energies_lat_mean, cov_inv, energies_exp - energies_lat_mean
            )
            print("r mag:\t", "\t".join(f"{100 * s:.5f}%" for s in [tmp[i].sum() / chi2 for i in range(n_levels)]))
        return chi2

    def get_chi2_resampling(self, p=None, cov_debugging=None, verbose=False):
        """
        get chi2 between the expected energy levels from the K matrix parameterization, using resampling data points.
            cov_debugging: covariance matrix for debugging, not recommended for real fit.
        """
        if p is not None:
            self.scattering_matrix.set_parameters(p)
        if self.scattering_matrix._p is None:
            raise ValueError("ScatteringMatrix parameters is not set yet, set_parameters(p) first.")
        if self.resampling_energies is None:
            raise ValueError("resampling_energies is not set yet.")
        n_levels = self.n_levels

        # energies_at_zeros = self.get_quantization_determinant_zeros(visiable=False)

        cov_inv = self.cov_inv
        if cov_debugging is not None:
            cov_inv = np.linalg.inv(cov_debugging)

        para_prior_dict = self.scattering_matrix.get_parameters()
        para_prior_list = list(para_prior_dict.values())
        para_keys = list(para_prior_dict.keys())

        chi2_resampling = np.zeros(self.n_resampling, "f8")
        parametrs_resampling = np.zeros((self.n_resampling, len(para_prior_list)), "f8")
        energies_exp_resampling = np.zeros((self.n_resampling, n_levels))

        for i_resamping in trange(self.n_resampling):
            energies_lat_r = self.resampling_energies[:, i_resamping]

            def minimum_funtion(input_para):
                energies_exp_r = np.zeros(n_levels)
                self.scattering_matrix.set_parameters(dict(zip(para_keys, input_para)))
                energies_at_zeros_r = self.get_quantization_determinant_zeros_resampling(
                    visiable=False, i_resampling=i_resamping
                )
                if len(energies_at_zeros_r) == 0:
                    energies_exp_r[:] = 0
                else:
                    for ie in range(n_levels):
                        idx = np.argmin(np.abs(energies_lat_r[ie] - energies_at_zeros_r))
                        energies_exp_r[ie] = energies_at_zeros_r[idx]
                chi2 = np.einsum("i, ij, j", energies_exp_r - energies_lat_r, cov_inv, energies_exp_r - energies_lat_r)
                # if verbose:
                #     print("E_exp: \t", energies_exp * self.at_inv_GeV)
                #     print(
                #         "E_lat: \t",
                #         gv.gvar(energies_lat.mean(axis=1), energies_lat.std(axis=1) * self.resampling_factor**0.5)
                #         * self.at_inv_GeV,
                #     )
                return chi2

            minimize_result = minimize(minimum_funtion, para_prior_list, method="Nelder-Mead")
            print("Optimization result:", minimize_result)
            print("Minimized chi2:", minimize_result.fun)
            print("Optimized parameters:", minimize_result.x)
            chi2_resampling[i_resamping] = minimize_result.fun
            parametrs_resampling[i_resamping] = minimize_result.x
            # get the energies expected from the optimized parameters.
            energies_exp_r = np.zeros(n_levels)
            self.scattering_matrix.set_parameters(dict(zip(para_keys, minimize_result.x)))
            energies_at_zeros_r = self.get_quantization_determinant_zeros_resampling(
                visiable=False, i_resampling=i_resamping
            )
            if len(energies_at_zeros_r) == 0:
                energies_exp_r[:] = 0
            if len(energies_at_zeros_r) == 0:
                energies_exp_r[:] = 0
            else:
                for ie in range(n_levels):
                    idx = np.argmin(np.abs(energies_lat_r[ie] - energies_at_zeros_r))

        return chi2_resampling, parametrs_resampling, energies_exp_resampling

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

    def plot_quantization_determinant(self, s, m1_A, m1_B, m2_A, m2_B, x):
        """
        plot to check behavior.
        """
        determinant = self.get_quantization_determinant(s, m1_A, m1_B, m2_A, m2_B, xi1=self.xi1_mean, xi2=self.xi2_mean)
        import matplotlib.pyplot as plt

        plt.plot(x, determinant, "-b")
        plt.ylim(-1, 1)
        plt.show()
        plt.clf()
