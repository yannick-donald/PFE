# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats.distributions import norm, lognorm, rv_frozen
import matplotlib.pyplot as plt


class BrownianMotion:
    """Brownian Motion (Wiener Process) with optional drift."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, t: np.ndarray, n: int, rnd: np.random.RandomState) -> np.ndarray:
        assert t.ndim == 1, "One dimensional time vector required"
        assert t.size > 0, "At least one time point is required"
        dt = np.concatenate((t[0:1], np.diff(t)))
        assert (dt >= 0).all(), "Increasing time vector required"
        # transposed simulation for automatic broadcasting
        W = rnd.normal(size=(n, t.size))
        W_drift = (W * np.sqrt(dt) * self.sigma + self.mu * dt).T
        return np.cumsum(W_drift, axis=0)

    def distribution(self, t: float) -> rv_frozen:
        return norm(self.mu * t, self.sigma * np.sqrt(t))


class GeometricBrownianMotion:
    """Geometric Brownian Motion.(with optional drift)."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, t: np.ndarray, n: int, rnd: np.random.RandomState) -> np.ndarray:
        assert t.ndim == 1, "One dimensional time vector required"
        assert t.size > 0, "At least one time point is required"
        dt = np.concatenate((t[0:1], np.diff(t)))
        assert (dt >= 0).all(), "Increasing time vector required"
        # transposed simulation for automatic broadcasting
        dW = (rnd.normal(size=(t.size, n)).T * np.sqrt(dt)).T
        W = np.cumsum(dW, axis=0)
        return np.exp(self.sigma * W.T + (self.mu - self.sigma ** 2 / 2) * t).T

    def distribution(self, t: float) -> rv_frozen:
        mu_t = (self.mu - self.sigma ** 2 / 2) * t
        sigma_t = self.sigma * np.sqrt(t)
        return lognorm(scale=np.exp(mu_t), s=sigma_t)


class BlackScholes:
    def __int__(self, S0: np.array, mu: np.array, sigma: np.ndarray, T: float):
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
        self.T = T

    def simulate(self, n: int, rnd: np.random.RandomState) -> np.ndarray:
        t = np.linspace(0, self.T)
        assert t.ndim == 1, "One dimensional time vector required"
        assert t.size > 0, "At least one time point is required"
        dt = np.concatenate((t[0:1], np.diff(t)))
        assert (dt >= 0).all(), "Increasing time vector required"
        # transposed simulation for automatic broadcasting
        dW = (rnd.normal(size=(t.size, n)).T * np.sqrt(dt)).T
        W = np.cumsum(dW, axis=0)
        return self.S0 * np.exp(self.sigma * W.T + (self.mu - self.sigma ** 2 / 2) * t).T

    import numpy as np


def simulate_stock_price_path(S0, mu, sigma, T, n, k):
    # Initialize variables
    a = mu - 0.5 * np.sum(sigma ** 2, axis=1)
    delta_t = T / k
    S_path = np.zeros((n, k + 1))

    # Set initial stock prices
    S_path[:, 0] = S0

    # Simulate stock price path
    for j in range(1, k + 1):
        delta = delta_t
        Z = np.random.normal(0, np.sqrt(delta), n)

        for i in range(n):
            S_path[i, j] = S_path[i, j - 1] * np.exp(a[j] * delta + np.sum(sigma[i] * Z))

    # Interpolate linearly
    t = np.linspace(0, T, k + 1)
    S_path = np.column_stack((t, S_path))

    return S_path


if __name__ == '__main__':
    # S = GeometricBrownianMotion(mu=2, sigma=0.3)
    # simulation = S.simulate(t=np.linspace(0, 3), n=3, rnd=np.random.RandomState(10))
    # print(simulation)

    # Example usage
    S0 = 100
    mu = np.array([0.05, 0.03, 0.02])
    sigma = np.array([[0.2, 0.0, 0.0],
                      [0.0, 0.1, 0.0],
                      [0.0, 0.0, 0.15]])
    T = 1
    n = 3
    k = 100

    S = simulate_stock_price_path(S0, mu, sigma, T, n, k)
    print(S)

    # Plot the option prices
    plt.plot(np.linspace(0, T), S, )
    plt.xlabel('Stock Price')
    plt.ylabel('Option Price')
    plt.title('Max Call Option Price using Linear Regression')
    plt.grid(True)
    plt.show()
