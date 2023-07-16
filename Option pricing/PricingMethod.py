from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Optional
from Option import Option
from StochasticProcess import StochasticProcess


class PricingMethod(ABC):
    @abstractmethod
    def price(self, option: Option, process: StochasticProcess) -> float:
        pass
