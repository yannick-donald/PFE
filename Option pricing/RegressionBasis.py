from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Optional
from scipy.special import eval_chebyt, eval_laguerre


class RegressionBasis(ABC):
    @abstractmethod
    def evaluate(self, x: Union[float, np.ndarray]) -> np.ndarray:
        pass


class ChebyshevBasis(RegressionBasis):
    def __init__(self, degree):
        self.degree = degree

    def evaluate(self, x):
        return eval_chebyt(np.arange(self.degree + 1)[:, None], x)



class LaguerreBasis(RegressionBasis):
    def evaluate(self, x):
        return eval_laguerre(np.arange(self.degree + 1), x)
