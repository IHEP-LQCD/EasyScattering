from utils import DoubleChannelCalculator, ScatteringCalculatorABC


class JointDoubleChannelCalculator(ScatteringCalculatorABC):
    def __init__(self, calculator1: DoubleChannelCalculator, calculator2: DoubleChannelCalculator):
        self.calculator1 = calculator1
        self.calculator2 = calculator2

    def get_chi2(self, p, verbose=False):
        chi2_1 = self.calculator1.get_chi2(p, cov_debugging=None, verbose=verbose)
        chi2_2 = self.calculator2.get_chi2(p, cov_debugging=None, verbose=verbose)
        return chi2_1 + chi2_2

    def get_chi2_resampling(self, p, cov_debugging=None, verbose=False):
        return super().get_chi2_resampling(p, cov_debugging, verbose)
