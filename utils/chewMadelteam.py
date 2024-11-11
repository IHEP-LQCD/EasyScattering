import numpy as np
import cmath
from .base import ChewMadelstemForm

class ChewMadelstemZero(ChewMadelstemForm):
    """
    A zero Chew-Madelstem variable.
    """
    def get_chew_madstem_matrix(self, s):
        return 0

class ChewMadelstemEqualMass(ChewMadelstemForm):
    """
    Chew-Madelstem for 2 equal mass.
    Ref: 10.1103/PhysRevD.88.014501
    """
    def get_chew_madstem_matrix(self, s, M_square):
        # print(msq, s , - 2 / np.pi  - s/ np.pi * 2 * cmath.sqrt(msq-s) * cmath.acos(cmath.sqrt((msq-s)/ msq)) / s**1.5)
        # print(cmath.sqrt((msq-s)/ msq),
        # cmath.sqrt(msq-s),
        # s/ np.pi * 2 * cmath.sqrt(msq-s) * cmath.acos(cmath.sqrt((msq-s)/ msq)) / s**1.5)
        return (
            -2 / np.pi
            - s
            / np.pi
            * 2
            * cmath.sqrt(M_square - s)
            * cmath.acos(cmath.sqrt((M_square - s) / M_square))
            / s**1.5
        )


class ChewMadelstemUnequalMass(ChewMadelstemForm):
    """
    Chew-Madelstem for unequal mass m1, m2
    """
    def get_chew_madstem_matrix(self, s, m1, m2):
        a = cmath.sqrt((m1 + m2) ** 2 - s)
        b = cmath.sqrt((m1 - m2) ** 2 - s)
        return (
            -2
            / np.pi
            * (
                -a * b / s * cmath.log((a + b) / 2 / (m1 * m2))
                + (m1**2 - m2**2) / 2 / s
            )
        )

class ChewMadelstemUnequalMass_2(ChewMadelstemForm):
    """
    Chew-Madelstem for unequal mass m1, m2
    """
    def get_chew_madstem_matrix(self, s, m1, m2):
        """
        this method define anothor kind of ChewMandelstam variable for unequal mass m1, m2
        """
        a = cmath.sqrt(s - (m1 + m2) ** 2)
        b = cmath.sqrt(s - (m1 - m2) ** 2)
        im = a * b
        re = (-2 * im * cmath.log((a + b) / 2 / np.sqrt(m1 * m2))) / np.pi
        # print(re, im)
        return (re + 1j * im) / s
