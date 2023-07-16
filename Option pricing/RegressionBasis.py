from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Optional


class RegressionBasis(ABC):
    @abstractmethod
    def evaluate(self, x: Union[float, np.ndarray]) -> np.ndarray:
        pass
