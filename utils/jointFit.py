from utils import DoubleChannelCalculator, ScatteringCalculatorABC
from typing import List


class JointDoubleChannelCalculator(ScatteringCalculatorABC):
    def __init__(self, calculators: List[ScatteringCalculatorABC]):
        self.calculators = calculators

    def get_chi2(self, p, verbose=False, is_nearest_zeros=False):
        total_chi2 = 0
        for calculator in self.calculators:
            total_chi2 += calculator.get_chi2(p, cov_debugging=None, is_nearest_zeros=is_nearest_zeros, verbose=verbose)
        return total_chi2

    def get_chi2_resampling(self, p, cov_debugging=None, verbose=False):
        total_chi2 = 0
        for calculator in self.calculators:
            total_chi2 += calculator.get_chi2_resampling(p, cov_debugging, verbose)
        return total_chi2
