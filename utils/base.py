from abc import ABC, abstractmethod
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


class ChewMadelstemForm(ABC):
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


class ScatteringMatrixForm(ABC, Analyticity):
    """
    ABC class for scattering matrix form.
    Inherit this class to define your scattering matrix form of K matrix or K inv.
    """

    def __init__(self, p, chew_madstem: ChewMadelstemForm):
        self.set_parameters(p)
        self.chew_madstem = chew_madstem

    def set_parameters(self, p):
        self._p = p

    @abstractmethod
    def get_K_matrix(self, s): ...

    @abstractmethod
    def get_K_inv_matrix(self, s): ...

    def get_t_inv_matrix(self, s):
        return self.get_K_inv_matrix(s) + self.chew_madstem.get_chew_madstem_matrix(s)

    def get_t_matrix(self, s):
        return np.linalg.inv(self.get_t_inv_matrix(s))

    def get_S_matrix(self, s, m1_A, m1_B, m2_A, m2_B):
        rho_sqrt = np.diag(
            (
                self.sqrt_vectorized(self.rho(s, m1_A, m1_B)),
                self.sqrt_vectorized(self.rho(s, m2_A, m2_B)),
            )
        )
        return np.identity(2) - 2j * rho_sqrt @ self.get_t_matrix(s) @ rho_sqrt

    @staticmethod
    def get_phase_from_S_matrix(S_matrix):
        """
        extract parameters delta1, eta1, delta2, eta2 from S matrix.
        """
        delta1 = np.angle(S_matrix[0, 0], deg=True) / 2 % 180
        eta1 = np.abs(S_matrix[0, 0])
        delta2 = np.angle(S[1, 1], deg=True) / 2 % 180
        eta2 = np.abs(S_matrix[1, 1])
        # print(f"phase1 = {delta1}, eta1 = {eta1}")
        # print(f"phase2 = {delta2}, eta2 = {eta2}")
        return delta1, eta1, delta2, eta2
