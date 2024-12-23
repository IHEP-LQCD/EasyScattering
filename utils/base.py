from abc import ABC, abstractmethod
import warnings
import numpy as np
import cmath


# ABC
class Analyticity:
    """
    define some basic functions for scattering matrix analysis.
    """

    @staticmethod
    def sqrt_vectorized(s):
        return np.vectorize(cmath.sqrt)(s)

    @staticmethod
    def sqrt_rh(s):
        sqrts = cmath.sqrt(s)
        return sqrts if sqrts.imag > 0 else -sqrts

    @staticmethod
    def scattering_mom2(s, m_A, m_B):
        """
        get scattering momemtum k^2.
        """
        k2 = (s - (m_A - m_B) ** 2) * (s - (m_A + m_B) ** 2) / 4 / s
        return k2

    @staticmethod
    def check_unitarity(s, S_matrix):
        """
        check unitarity of S matrix.
        """
        isunitary = np.allclose(S_matrix @ S_matrix.conj().T, np.identity(2)) and np.allclose(
            S_matrix.conj().T @ S_matrix, np.identity(2)
        )
        if not isunitary:
            warnings.warn(
                f"at sqrts = {s**.5}, S matrix not unitary:\n SdagS =\n {S_matrix.conj().T @ S_matrix}\n SSdag=\n{S_matrix @ S_matrix.conj().T}",
                Warning,
            )
        return isunitary

    def scattering_mom(self, s, m_A, m_B):
        """
        get scattering momemtum k under right hand sqrt.
        """
        k2 = self.scattering_mom2(s, m_A, m_B)
        return np.vectorize(self.sqrt_rh)(k2)

    def rho(self, s, m_A, m_B):
        """
        define rho(s)
        """
        k = self.scattering_mom(s, m_A, m_B)
        return 2 * k / self.sqrt_vectorized(s)

    def rho_matrix(self, s, m1_A, m1_B, m2_A, m2_B):
        """
        define rho(s) 2 by 2 matrix.
        """

        def get_rho_matrix_fcn(s0, m1_A, m1_B, m2_A, m2_B):
            return np.diag([self.rho(s0, m1_A, m1_B), self.rho(s0, m2_A, m2_B)])

        if isinstance(s, np.ndarray):
            rho_matrix = np.array([get_rho_matrix_fcn(s0, m1_A, m1_B, m2_A, m2_B) for s0 in s])
        else:
            rho_matrix = get_rho_matrix_fcn(s, m1_A, m1_B, m2_A, m2_B)
        return rho_matrix

    def rho_heviside(self, s, m_A, m_B):
        """
        define rho(s) Theta(s - (m_A + m_B)^2 > 0)
        """
        k = self.scattering_mom(s, m_A, m_B).real
        return 2 * k / self.sqrt_vectorized(s)

    def rho_heviside_matrix(self, s, m1_A, m1_B, m2_A, m2_B):
        """
        define rho(s) 2 by 2 matrix.
        """

        def get_rho_heviside_matrix_fcn(s0, m1_A, m1_B, m2_A, m2_B):
            return np.diag([self.rho_heviside(s0, m1_A, m1_B), self.rho_heviside(s0, m2_A, m2_B)])

        if isinstance(s, np.ndarray):
            rho_heviside_matrix = np.array([get_rho_heviside_matrix_fcn(s0, m1_A, m1_B, m2_A, m2_B) for s0 in s])
        else:
            rho_heviside_matrix = get_rho_heviside_matrix_fcn(s, m1_A, m1_B, m2_A, m2_B)
        return rho_heviside_matrix


class ChewMadelstemABC(ABC):
    """
    ABC class for Chew-Madelstem matrix.
    """

    @abstractmethod
    def get_chew_madstem_matrix(self, s): ...

    @staticmethod
    def plot_ChewMandelstam_function(fcn):
        x = np.linspace(-2.5, 10.5, 1000)
        y = fcn(x)
        y_real, y_imag = y.real, y.imag
        import matplotlib.pyplot as plt

        plt.plot(x, y_real, "-bx", label="real")
        plt.plot(x, y_imag, "-rx", label="imag")
        plt.legend()
        plt.show()
        plt.clf()


class ScatteringMatrixABC(ABC, Analyticity):
    """
    ABC class for scattering matrix form.
    Inherit this class to define your scattering matrix form of K matrix or K inv.
    """

    def __init__(self, chew_madstem: ChewMadelstemABC):
        self.chew_madstem = chew_madstem
        self._p = None

    def set_parameters(self, p):
        self._p = p

    def get_parameters(self):
        return self._p

    @abstractmethod
    def get_K_matrix(self, s): ...

    @abstractmethod
    def get_K_inv_matrix(self, s): ...

    def get_t_inv_matrix(self, s, m1_A, m1_B, m2_A, m2_B):
        if self._p is None:
            raise ValueError("parameters not set, please set_parameters(para) before.")
        return self.get_K_inv_matrix(s) + self.chew_madstem.get_chew_madstem_matrix(s, m1_A, m1_B, m2_A, m2_B)

    def get_t_matrix(self, s, m1_A, m1_B, m2_A, m2_B):
        if self._p is None:
            raise ValueError("parameters not set, please set_parameters(para) before.")
        return np.linalg.inv(self.get_t_inv_matrix(s, m1_A, m1_B, m2_A, m2_B))

    def get_rhot_square_abs(self, s_array, m1_A, m1_B, m2_A, m2_B):
        t_matrix = self.get_t_matrix(s_array, m1_A, m1_B, m2_A, m2_B)
        rho_sqrt_matrix = self.sqrt_vectorized(self.rho_heviside_matrix(s_array, m1_A, m1_B, m2_A, m2_B))
        rhot_square_abs = np.abs(rho_sqrt_matrix @ t_matrix @ rho_sqrt_matrix) ** 2
        return rhot_square_abs

    def get_S_matrix_from_t(self, s, m1_A, m1_B, m2_A, m2_B):
        def get_S_matrix_fcn(s0):
            rho_sqrt = np.diag(
                [
                    self.sqrt_vectorized(self.rho(s0, m1_A, m1_B)),
                    self.sqrt_vectorized(self.rho(s0, m2_A, m2_B)),
                ]
            )
            S_matrix_ret = np.identity(2) + 2j * rho_sqrt @ self.get_t_matrix(s0, m1_A, m1_B, m2_A, m2_B) @ rho_sqrt
            self.check_unitarity(s0, S_matrix_ret)
            return S_matrix_ret

        if isinstance(s, np.ndarray):
            S_matrix = np.array([get_S_matrix_fcn(s0) for s0 in s])
        else:
            S_matrix = get_S_matrix_fcn(s)
        return S_matrix

    def get_S_matrix_from_K(self, s):
        def get_S_matrix_fcn(s0):
            K = self.get_K_matrix(s0)
            S_matrix_ret = (np.identity(2) + 1j * K) * np.linalg.inv(np.identity(2) - 1j * K)
            return S_matrix_ret

        if isinstance(s, np.ndarray):
            S_matrix = np.array([get_S_matrix_fcn(s0) for s0 in s])
        else:
            S_matrix = get_S_matrix_fcn(s)
        return S_matrix

    @staticmethod
    def get_phase_from_S_matrix(S_matrix):
        """
        extract parameters delta1, eta1, delta2, eta2 from S matrix.
        """

        delta1 = (np.angle(S_matrix[0, 0], deg=True)) / 2 % 180
        delta2 = (np.angle(S_matrix[1, 1], deg=True)) / 2 % 180
        eta1 = np.abs(S_matrix[0, 0])
        eta2 = np.abs(S_matrix[1, 1])
        # print(f"phase1 = {delta1}, eta1 = {eta1}")
        # print(f"phase2 = {delta2}, eta2 = {eta2}")
        return delta1, eta1, delta2, eta2

    @staticmethod
    def plot_phase(S_matrix_array, x):
        delta1, eta1, delta2, eta2 = (
            np.zeros_like(x),
            np.zeros_like(x),
            np.zeros_like(x),
            np.zeros_like(x),
        )
        for i, s in enumerate(S_matrix_array):
            delta1[i], eta1[i], delta2[i], eta2[i] = ScatteringMatrixABC.get_phase_from_S_matrix(s)
        # print(s)
        # exit()
        import matplotlib.pyplot as plt

        plt.plot(x, delta1, "bx", label="delta1")
        plt.plot(x, delta2, "rx", label="delta2")
        plt.legend()
        plt.show()
        plt.clf()

        plt.plot(x, eta1, "bx", label="eta1")
        plt.plot(x, eta2, "rx", label="eta2")
        plt.ylim(0, 1.2)
        plt.legend()
        plt.show()
        plt.clf()

    def plot_rhot_square(self, s_array, m1_A, m1_B, m2_A, m2_B, x):
        rhot_square = self.get_rhot_square_abs(s_array, m1_A, m1_B, m2_A, m2_B)
        import matplotlib.pyplot as plt

        plt.plot(x, rhot_square[:, 0, 0], "bx", label="|T|^2")
        plt.plot(x, rhot_square[:, 1, 1], "rx", label="|T|^2")
        plt.ylim(0, 1.2)
        plt.legend()
        plt.show()
        plt.clf()


class ScatteringCalculatorABC(ABC, Analyticity):
    """
    ABC class for scattering matrix calculator.
    Inherit this class to define your scattering matrix calculator.
    """

    @abstractmethod
    def __init__(self, Ls, Q, cut, at_inv_GeV):
        self.Ls = Ls
        self.Q = Q
        self.cut = cut
        self.at_inv_GeV = at_inv_GeV
        self._scattering_matrix = None
        self._resampling_energies = None
        self._resampling_input = None

    # @abstractmethod
    # def set_scattering_matrix(self, scattering_matrix: ScatteringMatrixForm):
    #     pass

    # @abstractmethod
    # def set_resampling_energies(self, energies, resampling_type="jackknife"):
    #     if resampling_type == "jackknife":
    #         self._resampling_energies = energies
    #     else:
    #         raise ValueError("resampling_type not supported.")
    #     pass

    @abstractmethod
    def get_chi2(self, p, cov_debugging=None, verbose=False):
        if self._scattering_matrix is None:
            raise ValueError("scattering_matrix not set, please set_scattering_matrix(scattering_matrix) before.")
        if self._resampling_energies is None:
            raise ValueError("resampling_energies not set, please set_resampling_energies(energies) before.")
        if self._resampling_input is None:
            raise ValueError("resampling_input not set, please set_resampling_input(input) before.")
        pass
