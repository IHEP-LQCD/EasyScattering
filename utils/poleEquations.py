from .base import Analyticity, ChewMadelstemForm, ScatteringMatrixForm
import numpy as np
import sympy as sp
import cmath
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

    def get_pole_positions(self, m1_A, m1_B, m2_A, m2_B):
        """
        TODO:
        get pole positions of scattering matrix.
        """
        if self.scattering_matrix is None:
            raise ValueError("Scattering matrix is not set, please set .set_scattering_matrix() first.")
        if self.scattering_matrix._p is None:
            raise ValueError("parameters not set, please scattering_matrix.set_parameters(para) before.")

        s = sp.symbols('s')
        p = sp.symbols('p', imaginary=True)  # p is treated as imaginary in the equation here
        n_chan = 2
        K_matrix = self.scattering_matrix.get_K_matrix(s)
        print(K_matrix)
        for ichan in range(n_chan):
            ma = m1_A if ichan == 0 else m2_A
            mb = m1_B if ichan == 0 else m2_B

            K = K_matrix[ichan, ichan]

            eq = sp.Eq(sp.sqrt(s) / K / 2 , sp.I * p)

            p_expr = (sp.sqrt_rh((s - (ma + mb)**2)) * cmath.sqrt(s - (ma - mb)**2)) / (2 * sp.sqrt(s))
            solutions = sp.solve(p_expr, s)

            # Extract solutions and return them
            return [solution.evalf() for solution in solutions]

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

        n_chan = 2
        residue = np.zeros(n_chan, dtype="c16")
        for ichan in range(n_chan):
            def integrand(t):
                return t_matrix(contour(t))[ichan, ichan] * d_contour(t)

            residue[ichan] = (
                integrate.quad(lambda t: np.real(integrand(t)), 0, 2 * np.pi)[0]
                + 1j * integrate.quad(lambda t: np.imag(integrand(t)), 0, 2 * np.pi)[0]
            ) / (2 * np.pi * 1j)

            if np.abs(residue[ichan]) < 1e-5:
                warnings.warn(f"Channel {ichan}, please check the pole position.")

        return -residue
