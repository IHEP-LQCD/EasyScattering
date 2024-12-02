from .base import Analyticity, ChewMadelstemForm, ScatteringMatrixForm
import numpy as np
from scipy import integrate
from functools import partial

import warnings


class PoleEquationsSolver(Analyticity):
    def __init__(self):
        pass

    def set_scattering_matrix(self, scattering_matrix: ScatteringMatrixForm):
        self.scattering_matrix = scattering_matrix

    def check_is_t_matrix_pole(self, s0, m1_A, m1_B, m2_A, m2_B):
        # t_matrix poles are zeros of t_inv_matrix
        t_inv = partial(self.scattering_matrix.get_t_inv_matrix, m1_A=m1_A, m1_B=m1_B, m2_A=m2_A, m2_B=m2_B)
        assert np.abs(t_inv(s0)) < 1e-5

    def get_pole_positions(self):
        """
        TODO:
        get pole positions of scattering matrix.
        """
        if self.scattering_matrix is None:
            raise ValueError("Scattering matrix is not set, please set .set_scattering_matrix() first.")
        if self.scattering_matrix._p is None:
            raise ValueError("parameters not set, please scattering_matrix.set_parameters(para) before.")
        pass

    def get_t_matrix_pole_residues(self, pole_s0, m1_A, m1_B, m2_A, m2_B):
        """
        get pole residues of scattering matrix.
            s0: pole position.
        """
        if self.scattering_matrix is None:
            raise ValueError("Scattering matrix is not set, please set .set_scattering_matrix() first.")
        if self.scattering_matrix._p is None:
            raise ValueError("parameters not set, please scattering_matrix.set_parameters(para) before.")

        # t_matrix poles are zeros of t_inv_matrix
        # check pole first
        # self.check_is_t_matrix_pole(pole_s0, m1_A=m1_A, m1_B=m1_B, m2_A=m2_A, m2_B=m2_B)
        t_matrix = partial(self.scattering_matrix.get_t_matrix, m1_A=m1_A, m1_B=m1_B, m2_A=m2_A, m2_B=m2_B)

        radio_contour = 1e-6

        def contour(t):
            return pole_s0 + radio_contour * np.exp(1j * t)

        def d_contour(t):
            return radio_contour * 1j * np.exp(1j * t)

        Nchan = 2
        for ichan in range(Nchan):
            def integrand(t):
                return t_matrix(contour(t))[ichan, ichan] * d_contour(t)

            residue = (
                integrate.quad(lambda t: np.real(integrand(t)), 0, 2 * np.pi)[0]
                + 1j * integrate.quad(lambda t: np.imag(integrand(t)), 0, 2 * np.pi)[0]
            ) / (2 * np.pi * 1j)
            print(residue)

        if np.abs(residue) < 1e-5:
            warnings.warn("residue is too small, please check the pole position.")

        return -residue
