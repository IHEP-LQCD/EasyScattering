import numpy as np
import math
import cmath
from math import fsum
from itertools import product
from scipy import special
from typing import List


class ScatteringSingleChannelCalculator:
    """
    A simple methods class to calculate scattering amplitude.
        - Only for Single Channel.
        - support fort both center-of-mass / rest frame and laboratory / moving frame.
        - design to support for ( L>=0 ) partial-wave (S,P,D...), but current only support for S-wave.
    Reference: https://link.aps.org/doi/10.1103/PhysRevD.85.114507

    Attributes:
        L (int): The system total anglar momentum.
        s (numpy.ndarray): Mandstem variable s.
        p (numpy.ndarray): total Momentum in the system.

    Methods:
        conv2CM(mass1, mass2, pa, pb, E_L_obs):
            Converts the observed laboratory energy E_L_obs to the energy and momentum in the CM system.

        E_CM_from_thres(mass1, mass2, pa, pb):
            Computes the energy and momentum in the CM system from threshold energy.

        factory_pcotdelta_fast_restframe(l1, m1, l2, m2):
            Generates a pcotdelta function for S-wave in the rest frame.

        factory_pcotdelta_fast_flight(l1, m1, l2, m2):
            Generates a pcotdelta function for S-wave in the flight frame.
    """

    L = None
    s = None
    p = None

    def __init__(self,
                 L: int,
                 cut_off: int = 30,
                #  a_s=0.1517,
                 xi_0=5.3,
                 a_t_inv_GeV=6.894) -> None:
        self.L = L
        self.d = None  #np.asarray([0, 0, 0])
        self.s = None
        self.gamma = None
        # gamma = 1.0 for rest frame
        # self.a_s = a_s
        self.a_t_inv_GeV = a_t_inv_GeV
        self.xi_0 = xi_0

        self.cut_off = cut_off if cut_off >= 30 else 30

        # self.fcn_M = self.factory_matrix_M(0, 0, 0, 0)

    def conv2CM(self, mass1: float, mass2: float, pa: List[int], pb: List[int],
                E_L_obs):
        pa = np.asarray(pa)
        pb = np.asarray(pb)
        self.d = pa + pb
        p2fac = (2 * np.pi / self.L * self.a_t_inv_GeV / self.xi_0)**2
        print(F"{p2fac} {np.sum(self.d**2)} {E_L_obs**2} {np.sum(self.d**2) * p2fac / E_L_obs**2}")
        self.gamma = 1 / np.sqrt(1 - np.sum(self.d**2) * p2fac / E_L_obs**2)
        E_CM_obs = np.sqrt(E_L_obs**2 - np.sum(self.d**2) * p2fac)
        # print(F"E_lab_obs = {E_L_obs}")
        p_CM_sq = (E_CM_obs**4 + mass1**4 + mass2**4 -
                   2 * mass1**2 * mass2**2 - 2 * E_CM_obs**2 * mass1**2 -
                   2 * E_CM_obs**2 * mass2**2) / (2 * E_CM_obs)**2
        E_L_thres = np.sqrt(np.sum(pa**2)*p2fac + mass1**2) + \
            np.sqrt(np.sum(pb**2)*p2fac + mass2**2)
        print(F"pa = {pa}, pb = {pb}")
        print(
            F"E_lab_obs = {E_L_obs}, E_lab_thres = {E_L_thres}, E_cm_obs = {E_CM_obs},  k*2 = {p_CM_sq / p2fac}, gamma = {self.gamma}"
        )
        self.s = self.d * (1 + (mass1**2 - mass2**2) / E_CM_obs**2)
        return E_CM_obs, p_CM_sq

    def E_CM_from_thres(
        self,
        mass1: float,
        mass2: float,
        pa: List[int],
        pb: List[int],
    ):
        # m1 = 0.350
        # m2 = 3.099
        pa = np.asarray(pa)
        pb = np.asarray(pb)
        d = pa + pb
        p2fac = (2 * np.pi / self.L * self.a_t_inv_GeV / self.xi_0)**2
        E_L_thres = np.sqrt(np.sum(pa**2)*p2fac + mass1**2) + \
            np.sqrt(np.sum(pb**2)*p2fac + mass2**2)
        gamma = 1 / np.sqrt(1 - np.sum(d**2) * p2fac / E_L_thres**2)
        E_CM_from_thres = np.sqrt(E_L_thres**2 - np.sum(d**2) * p2fac)
        p_CM_sq = (
            E_CM_from_thres**4 + mass1**4 + mass2**4 -
            2 * mass1**2 * mass2**2 - 2 * E_CM_from_thres**2 * mass1**2 -
            2 * E_CM_from_thres**2 * mass2**2) / (2 * E_CM_from_thres)**2
        q2 = p_CM_sq/p2fac
        # print(F"pa = {pa}, pb = {pb}")
        # print(
        #     F"E_lab_thres = {E_L_thres}, E_cm_from_thres = {E_CM_from_thres},  k*2 = {q2}, gamma = {gamma}"
        # )
        return E_CM_from_thres, q2

    # @staticmethod
    def factory_pcotdelta_fast_restframe(self, l1: int, m1: int, l2: int,
                                         m2: int):
        if l1 != 0 or l2 != 0:
            raise ValueError("Undo: ONLY support for S-wave!!!")
        xi_0 = self.xi_0
        a_t_inv_GeV = self.a_t_inv_GeV
        L = self.L
        unitFac = xi_0 / a_t_inv_GeV

        Z3_cut = self.cut_off
        Z3_space = np.array(list(product(range(-Z3_cut, Z3_cut + 1),
                                         repeat=3)),
                            dtype="f8")
        gamma = 1
        gamma_inv = 1 / gamma
        Z3_cut2 = Z3_cut**2

        tmp2 = np.einsum("ab,ab->a", Z3_space, Z3_space)
        sum_indi = (tmp2 < Z3_cut2)
        # sum_indi = np.ones_like(tmp2, dtype=bool)
        sum_indi[tmp2 == 0] = False

        n = Z3_space[sum_indi]
        # sn = np.einsum("ab, b->a", n, s)
        phase_fac = 1  #np.exp(np.pi * 1j * sn)
        n2 = np.einsum("ab, ab ->a", n, n)
        n_abs = np.sqrt(np.einsum("ab, ab ->a", n, n))

        # z = n  #- np.einsum("a, b ->ab", (0.5 + (gamma-1) * s2_inv * sn) * gamma_inv,  s)
        # z2 = np.einsum("ab, ab ->a", z, z)
        # w = n #- np.einsum("a, b ->ab", (1 - gamma) * s2_inv * sn,  s)
        # w_abs = np.sqrt(np.einsum("ab, ab ->a", w, w))

        z_0_sq = 0

        def pcotdelta(k2):
            # q2 = (k * L * a_s / 2 / np.pi)**2 # Note: GeV * fm
            # k = np.real(k)
            lam = 1
            q2 = (L * unitFac / 2 / np.pi)**2 * k2  # Note: GeV * fm
            # print(f"q2    : {q2}")
            c_minus_q2 = cmath.sqrt(-q2 * lam)
            n_abs_lam = n_abs / np.sqrt(lam)
            sum0 = np.exp(lam * (q2-n2)) / (n2-q2) / np.sqrt(4 * np.pi) + \
                        gamma * np.sqrt(np.pi)/4 * (np.exp(-2 * np.pi * n_abs_lam*c_minus_q2) *\
                                                        (2 - special.erfc(c_minus_q2 - np.pi * n_abs_lam) + \
                                                            np.exp(4 * np.pi * c_minus_q2 * n_abs_lam) * special.erfc(c_minus_q2 + np.pi * n_abs_lam)
                                                        ) \
                                                    ) / n_abs_lam * phase_fac / np.sqrt(lam)
            Zd00 = fsum(sum0.real)
            assert fsum(sum0.imag) < 1e-5
            Zd00 += np.exp(lam *
                           (q2 - z_0_sq)) / (z_0_sq - q2) / np.sqrt(4 * np.pi)
            if k2 >= 0:
                Zd00 += gamma * np.pi * (np.exp(lam * q2) * (2 * np.sqrt(
                    lam * q2) * special.dawsn(np.sqrt(lam * q2)) - 1))
            else:
                Zd00 += gamma * np.pi * (np.exp(lam * q2) - np.sqrt(
                    -np.pi * lam * q2) * special.erf(np.sqrt(-lam * q2)))
            pcotdelta0 = 2 * Zd00 / (np.sqrt(np.pi) * L * xi_0 / a_t_inv_GeV *
                                     gamma)
            # print(f"pcotdelta = {pcotdelta0}")
            return pcotdelta0

        return (pcotdelta)

    def factory_pcotdelta_fast_flight(self, l1: int, m1: int, l2: int,
                                      m2: int):
        if l1 != 0 or l2 != 0:
            raise ValueError("Undo: ONLY S-wave!!!")
        xi_0 = self.xi_0
        a_t_inv_GeV = self.a_t_inv_GeV
        L = self.L
        s = np.asarray(self.s)
        unitFac = xi_0 / a_t_inv_GeV
        s2 = np.dot(s, s)
        s2_inv = 1 if s2 == 0 else 1 / np.dot(s, s)

        Z3_cut = self.cut_off
        Z3_space = np.array(list(product(range(-Z3_cut, Z3_cut + 1),
                                         repeat=3)),
                            dtype="f8")
        gamma = self.gamma
        gamma_inv = 1 / gamma
        Z3_cut2 = Z3_cut**2

        tmp2 = np.einsum("ab,ab->a", Z3_space, Z3_space)
        sum_indi = (tmp2 < Z3_cut2)
        # sum_indi = np.ones_like(tmp2, dtype=bool)
        sum_indi[tmp2 == 0] = False

        n = Z3_space[sum_indi]
        sn = np.einsum("ab, b->a", n, s)
        phase_fac = np.exp(np.pi * 1j * sn)
        z = n - np.einsum("a, b ->ab",
                          (0.5 + (gamma - 1) * s2_inv * sn) * gamma_inv, s)
        z2 = np.einsum("ab, ab ->a", z, z)
        w = n - np.einsum("a, b ->ab", (1 - gamma) * s2_inv * sn, s)
        w_abs = np.sqrt(np.einsum("ab, ab ->a", w, w))

        z_0_sq = (gamma_inv * 0.5)**2 * s2

        def pcotdelta(k2, lam: float = 1):
            # q2 = (k * L * a_s / 2 / np.pi)**2 # Note: GeV * fm
            # k = np.real(k)
            q2 = (L * unitFac / 2 / np.pi)**2 * k2  # Note: GeV * fm
            # print(f"q2    : {q2}")
            c_minus_q2 = cmath.sqrt(-q2 * lam)
            w_abs_lam = w_abs / np.sqrt(lam)
            term1 = np.exp(lam * (q2 - z2)) / (z2 - q2) / np.sqrt(4 * np.pi)
            term2 = gamma / np.sqrt(lam) * np.sqrt(np.pi) / 4 *\
                    (np.exp(-2 * np.pi * w_abs_lam  * c_minus_q2) *\
                        (2 - special.erfc(c_minus_q2 - np.pi * w_abs_lam ) + \
                            np.exp(4 * np.pi * c_minus_q2 * w_abs_lam) * special.erfc(c_minus_q2 + np.pi * w_abs_lam)
                        ) \
                    ) / w_abs_lam * phase_fac
            sum0 = term1 + term2
            Zd00 = fsum(sum0.real)
            if fsum(sum0.imag) > 1e-5:
                raise ValueError(F"imag sum divergence, s={s}, gamma={gamma}")
            term3 = np.exp(lam *
                           (q2 - z_0_sq)) / (z_0_sq - q2) / np.sqrt(4 * np.pi)
            Zd00 += term3
            if k2 >= 0:
                term4 = gamma * np.pi * (np.exp(q2 * lam) * (2 * np.sqrt(
                    q2 * lam) * special.dawsn(np.sqrt(q2 * lam)) - 1))
            else:
                term4 = gamma * np.pi * (np.exp(q2 * lam) - np.sqrt(
                    -np.pi * q2 * lam) * special.erf(np.sqrt(-q2 * lam)))
            Zd00 += term4
            pcotdelta0 = Zd00 * 2 / (np.sqrt(np.pi) * L * xi_0 / a_t_inv_GeV *
                                     gamma)
            term1 = fsum(term1.real)
            term2 = fsum(term2.real)
            term1 *= 2 / (np.sqrt(np.pi) * L * xi_0 / a_t_inv_GeV * gamma)
            term2 *= 2 / (np.sqrt(np.pi) * L * xi_0 / a_t_inv_GeV * gamma)
            term3 *= 2 / (np.sqrt(np.pi) * L * xi_0 / a_t_inv_GeV * gamma)
            term4 *= 2 / (np.sqrt(np.pi) * L * xi_0 / a_t_inv_GeV * gamma)

            # print(f"\n{term1}\t{(term2)}\t{term3}\t{term4}")
            return pcotdelta0

        return (pcotdelta)
