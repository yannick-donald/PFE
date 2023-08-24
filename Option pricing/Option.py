from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Optional

from scipy.stats import gmean


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


class MaxCallOption(Option):
    def __init__(self, K, T, S0):
        self.K = K
        self.T = T
        self.S0 = S0

    def payoff(self, S):
        return np.maximum(np.max(S, axis=1) - self.K, 0)

    def __str__(self):
        return f"MaxCallOption ({self.K, self.T, self.S0})"


class MinPutOption(Option):
    def __init__(self, K, T, S0):
        self.K = K
        self.T = T
        self.S0 = S0

    def payoff(self, S):
        return np.maximum(self.K - np.min(S, axis=1), 0)

    def __str__(self):
        return f"MinPutOption ({self.K, self.T, self.S0})"


class MaxPutOption(Option):
    def __init__(self, K, T, S0):
        self.K = K
        self.T = T
        self.S0 = S0

    def payoff(self, S):
        return np.maximum(self.K - np.max(S, axis=1), 0)

    def __str__(self):
        return f"MaxPutOption ({self.K, self.T, self.S0})"


class ArithmeticPutOption(Option):
    def __init__(self, K, T, S0):
        self.K = K
        self.T = T
        self.S0 = S0

    def payoff(self, S):
        return np.maximum(self.K - np.mean(S, axis=1), 0)

    def __str__(self):
        return f"ArithmeticPutOption ({self.K, self.T, self.S0})"


class GeometricPutOption(Option):
    def __init__(self, K, T, S0):
        self.K = K
        self.T = T
        self.S0 = S0

    def payoff(self, S):
        return np.maximum(self.K - gmean(S, axis=1), 0)

    def __str__(self):
        return f"GeometricPutOption ({self.K, self.T, self.S0})"


class CallOption(Option):

    def __init__(self, K, T, S0):
        self.K = K
        self.T = T
        self.S0 = S0

    def payoff(self, S: np.ndarray) -> np.ndarray:
        return np.maximum(S - self.K, 0)

    def __str__(self):
        return f"CallOption ({self.K, self.T, self.S0})"


class PutOption(Option):

    def __init__(self, K, T, S0):
        self.K = K
        self.T = T
        self.S0 = S0

    def payoff(self, S: np.ndarray) -> np.ndarray:
        return np.maximum(self.K - S, 0)

    def __str__(self):
        return f"PutOption ({self.K, self.T, self.S0})"
