from PricingMethod import PricingMethod

from Option import Option
from Option import Option
from StochasticProcess import StochasticProcess
from RegressionBasis import RegressionBasis


class LongstaffSchwartz(PricingMethod):
    def __init__(self, basis: RegressionBasis):
        self.basis = basis

    def price(self, option: Option, process: StochasticProcess) -> float:
        # To be implemented
        pass
