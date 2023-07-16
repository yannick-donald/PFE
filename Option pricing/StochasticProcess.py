from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Optional
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

class StochasticProcess(ABC):
    def __init__(self, dim: int, r: Union[float, np.ndarray], sigma: Union[float, np.ndarray]):
        self.dim = dim
        self.r = r
        self.sigma = sigma

    @abstractmethod
    def generate_paths(self, S0: Union[float, np.ndarray], N: int, m: int) -> np.ndarray:
        pass


class BlackScholesProcess(StochasticProcess):
    def __init__(self, dim: int, r: Union[float, np.ndarray], sigma: Union[float, np.ndarray],
                 mu: Union[float, np.ndarray]):
        super().__init__(dim, r, sigma)
        self.mu = mu

    def generate_paths(self, S0: Union[float, np.ndarray], N: int, m: int, T: float) -> np.ndarray:
        dt = T / m
        dW = np.random.normal(size=(N, m, self.dim)) * np.sqrt(dt)  # increments of standard Brownian motion
        W = np.cumsum(dW, axis=1)  # standard Brownian motion
        t = np.linspace(dt, T, m)  # time grid
        if self.dim > 1:
            L = cholesky(self.sigma, lower=True)  # Cholesky decomposition of the covariance matrix
            W = np.tensordot(W, L, axes=(-1, -1))  # correlated Brownian motion
        S = S0 * np.exp((self.mu - 0.5 * np.diag(self.sigma)) * t[:, None] + W)  # geometric Brownian motion
        return S


if __name__ == '__main__':
    # Parameters
    S0 = np.array([100, 100])
    N = 10000
    m = 100
    T = 1.0
    r = np.array([0.05, 0.05])
    sigma = np.array([[0.2 ** 2, 0.2 * 0.3 * 0.5], [0.2 * 0.3 * 0.5, 0.3 ** 2]])  # covariance matrix
    mu = np.array([0.1, 0.1])

    # Create Black-Scholes process
    process = BlackScholesProcess(dim=2, r=r, sigma=sigma, mu=mu)

    # Generate paths
    paths = process.generate_paths(S0, N, m, T)

    # Print some paths
    print(paths[:5, :, 0])  # paths for the first asset
    print(paths[:5, :, 1])  # paths for the second asset


    # Create a new figure
    plt.figure()

    # Plot the first 10 paths for the first asset
    for i in range(10):
        plt.plot(np.linspace(0, T, m), paths[i, :, 0])
    plt.title("10 sample paths for the first asset")
    plt.show()

    # Create another new figure
    plt.figure()

    # Plot the first 10 paths for the second asset
    for i in range(10):
        plt.plot(np.linspace(0, T, m), paths[i, :, 1])
    plt.title("10 sample paths for the second asset")
    plt.show()

