
import numpy as np
from .base import ScatteringMatrixForm

class KMatraixSumOfPoles(ScatteringMatrixForm):
    """
    __init__ method define by super class.
    """

    def get_K_matrix(self, s):
        p = self._p

        def K_matrix_fcn(s0):
            g = np.array(
                [
                    [p["g1"] * p["g1"], p["g1"] * p["g2"]],
                    [p["g1"] * p["g2"], p["g2"] * p["g2"]],
                ],
                dtype="f8",
            )
            gamma = np.array(
                [[p["gamma11"], p["gamma12"]], [p["gamma12"], p["gamma22"]]], dtype="f8"
            )
            return 1 / ((p["M^2"] - s0)) * g + gamma

        # K_matrix = (np.vectorize(K_matrix_fcn))(s)
        if isinstance(s, np.ndarray):
            K_matrix = np.array([K_matrix_fcn(s0) for s0 in s])
        else:
            K_matrix = K_matrix_fcn(s)
        # print(K_matrix)
        return K_matrix

    def get_K_inv_matrix(self, s):
        return np.linalg.inv(self.get_K_matrix(s))


class KinvMatraixPolymomial(ScatteringMatrixForm):
    """
    __init__ method define by super class.
    """

    def get_K_matrix(self, s):
        return np.linalg.inv(self.get_K_inv_matrix(s))

    def get_K_inv_matrix(self, s):
        p = self._p

        def K_matrix_fcn(s0):
            c0 = np.array(
                [
                    [p["c0_11"], p["c0_12"]],
                    [p["c0_12"], p["c0_22"]],
                ],
                dtype="f8",
            )
            c1 = np.array(
                [
                    [p["c1_11"], p["c1_12"]],
                    [p["c1_12"], p["c1_22"]],
                ],
                dtype="f8",
            )
            c2 = np.array(
                [
                    [p["c2_11"], p["c2_12"]],
                    [p["c2_12"], p["c2_22"]],
                ],
                dtype="f8",
            )

            return c0 + c1 * s0 + c2 * s0**2

        Kinv_matrix = np.array([K_matrix_fcn(s0) for s0 in s])
        return Kinv_matrix


class KinvMatraixPolymomialSqrts(ScatteringMatrixForm):
    """
    __init__ method define by super class.
    """

    def get_K_matrix(self, s):
        return np.linalg.inv(self.get_K_inv_matrix(s))

    def get_K_inv_matrix(self, s):
        p = self._p

        def K_matrix_fcn(s0):
            sqrt_s = np.sqrt(s0)
            c0 = np.array(
                [
                    [p["c0_11"], p["c0_12"]],
                    [p["c0_12"], p["c0_22"]],
                ],
                dtype="f8",
            )
            c1 = np.array(
                [
                    [p["c1_11"], p["c1_12"]],
                    [p["c1_12"], p["c1_22"]],
                ],
                dtype="f8",
            )
            c2 = np.array(
                [
                    [p["c2_11"], p["c2_12"]],
                    [p["c2_12"], p["c2_22"]],
                ],
                dtype="f8",
            )

            return c0 + c1 * sqrt_s + c2 * s0

        Kinv_matrix = np.array([K_matrix_fcn(s0) for s0 in s])
        return Kinv_matrix