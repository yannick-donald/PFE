from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Optional


class Option(ABC):
    def __init__(self, K: float, T: float):
        self.K = K
        self.T = T

    @abstractmethod
    def payoff(self, S: np.ndarray) -> np.ndarray:
        pass


class AmericanOption(Option):
    def __init__(self, K: float, T: float):
        super().__init__(K, T)
        self.early_exercise_boundary = None

    def price(self, process):
        # To be implemented
        pass

    def optimal_stopping_time(self, process):
        # To be implemented
        pass

    def early_exercise_boundary(self, process):
        # To be implemented
        pass
