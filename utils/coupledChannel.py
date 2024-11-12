import numpy as np
import cmath
from math import fsum
from scipy.interpolate import interp1d
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
        # self.at_inv_GeV = 6.894
        if xi_0 is None:
            xi_0 = 1.0
            warnings.warn("aspect_ratio xi_0 is not set, use default 1.0.", UserWarning)
        self.xi_0 = xi_0

        self.cut = cut if cut >= 30 else 30
        self.q2_begin = -1.0001
        self.q2_end = 9.0001
        self.q2_density = 100000  # 2**17
        self.cache_file_dir = cache_file_dir
        if not os.path.exists(cache_file_dir):
            os.makedirs(cache_file_dir)
        # self.fcn_M = self.factory_matrix_M(0, 0, 0, 0)
        self.init_M0000_cache()

    def init_M0000_cache(self):
        cache_file_name = f"{self.cache_file_dir}/cache_{self.q2_begin}_{self.q2_end}_{self.q2_density}_{self.cut}.npy"
        s = perf_counter()
        self.zeta_x = np.linspace(
            self.q2_begin, self.q2_end, self.q2_density, dtype="f8"
        )
        if os.path.exists(cache_file_name):
            self.zeta_y = np.load(cache_file_name)
        else:
            # fcn_M = self.factory_matrix_M(0, 0, 0, 0)
            fcn_M = self.kM0000_simpified
            self.zeta_y = fcn_M(self.zeta_x)
            np.save(cache_file_name, self.zeta_y)
        interpolator = interp1d(self.zeta_x, self.zeta_y, kind="linear")
        self.M0000 = interpolator
        print(
            f"INIT CACHE TIME: {perf_counter()-s:.3f} secs, file size = {self.zeta_y.nbytes/1024:.3f} KB"
        )

    def kM0000_simpified(self, q2):
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
        return ret / (self.Ls * np.pi) / (1 / self.xi_0)

    def set_scattering_matrix(self, form: ScatteringMatrixForm):
        """
        setter of scattering matrix form.
        """
        self.scattering_matrix = form

    def set_chew_mandelstam(self, form: ChewMadelstemForm = None):
        """
        settter of chew mandelstam form.
        """
        self.chew_mandelstam = form

    def set_energy_levels_data_jackknife(self, data):
        """
        setter of energy levels jackknife data.
            set data before set parameters of scattering matrix, then get the chi2.
        """
        self.n_levels, self.N_cfg = data.shape
        self.energy_levels_data = data

    # def set_zeta_function(self, zeta):
    #     pass

    @staticmethod
    def plot_zeta_function(fcn):
        x = np.linspace(-0.9, 8.5, 10000)
        y = fcn(x)
        import matplotlib.pyplot as plt

        plt.plot(x, y, "--b")
        plt.ylim(-10, 10)
        plt.show()
        plt.clf()

    def get_luescher_determint(self, s, m1_A, m1_B, m2_A, m2_B):
        '''
        Det [K^-1 - diag(rho1 M0000, rho2 M0000)] = 0.
        rho M0000 = 2 k / sqrt(s) M0000 = 2/sqrt(s) * kM0000
        '''
        K_inv = self.scattering_matrix.get_K_inv_matrix(s)

        # dimensionless: q2 = (k * L / 2 / np.pi) ** 2
        # a_s = aspect_ratio * a_t = aspect_ratio / at_inv_GeV
        q_square_1 = self.scattering_mom2(s, m1_A, m1_B) * (self.xi_0 / 2/ np.pi)**2
        q_square_2 = self.scattering_mom2(s, m2_A, m2_B) * (self.xi_0 / 2/ np.pi)**2
        rho_M0000_1 = 2 / np.vectorize(cmath.sqrt)(s) *  self.kM0000_simpified(q_square_1)
        rho_M0000_2 = 2 / np.vectorize(cmath.sqrt)(s) *  self.kM0000_simpified(q_square_2)
        return np.linalg.det(K_inv - np.diag((rho_M0000_1, rho_M0000_2)))

