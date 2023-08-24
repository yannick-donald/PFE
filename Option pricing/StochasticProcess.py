from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Optional
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

from scipy.stats import norm
from numpy.random import Generator, PCG64
import sobol_seq
from ghalton import Halton
from scipy.stats import qmc


class StochasticProcess(ABC):
    def __init__(self, dim: int, r: Union[float, np.ndarray]):
        self.dim = dim
        self.r = r

    @abstractmethod
    def generate_paths(self, S0: Union[float, np.ndarray], N: int, m: int) -> np.ndarray:
        pass


class BlackScholesProcess(StochasticProcess):

    def __init__(self, dim: int, r: Union[float, np.ndarray],
                 mu: Union[float, np.ndarray], vol: Union[float, np.ndarray], corr: Union[float, np.ndarray]):
        super().__init__(dim, r)
        self.mu = mu
        self.vol = vol
        self.corr = corr
        self.sigma = np.outer(vol, vol) * corr

    def generate_paths(self, S0: np.ndarray, N: int, m: int, T: float) -> np.ndarray:

        dt = T / m
        rng = np.random.default_rng()
        dW = rng.normal(size=(N, m, self.dim),) * np.sqrt(dt)  # increments of standard Brownian motion
        W = np.cumsum(dW, axis=1)  # standard Brownian motion

        if self.dim > 1:
            L = np.linalg.cholesky(self.corr).T
            W = np.einsum('nmi,ij->nmj', W, L)

        t = np.linspace(dt, T, m)  # time grid
        S = np.zeros((N, m + 1, self.dim))  # initialize S
        S[:, 0, :] = S0  # set initial asset prices

        # apply geometric Brownian motion formula for remaining times
        S[:, 1:, :] = S0[None, None, :] * np.exp(
            (self.mu[None, None, :] - 0.5 * self.vol[None, None, :] ** 2) * t[None, :, None] + self.vol[None, None,
                                                                                               :] * W)

        return S

    def generate_paths_qmc(self, S0: np.ndarray, N: int, m: int, T: float, type_qmc='sobol') -> np.ndarray:

        dt = T / m

        if type_qmc == 'sobol':
            dW = np.zeros((N, m, 3))

            # qmc_points = sobol_seq.i4_sobol_generate(self.dim * m, N)  # generate QMC points
            # qmc_points = norm.ppf(qmc_points)  # apply inverse standard normal CDF

            for i in range(self.dim):
                engine = qmc.Sobol(m, scramble=True)  # Sobol engine
                qmc_points = engine.random(N)  # generate QMC points
                qmc_points = norm.ppf(qmc_points)
                dW[:, :, i] = qmc_points





        elif type_qmc == 'halton':

            # sequencer = Halton(self.dim * m)  # Halton sequencer
            # qmc_points = sequencer.get(N)  # generate QMC points
            # qmc_points = norm.ppf(qmc_points)  # apply inverse standard normal CDF
            dW = np.zeros((N, m, self.dim))
            for i in range(self.dim):
                engine = qmc.Halton(m, scramble=True)  # Halton engine
                qmc_points = engine.random(N)  # generate QMC points
                qmc_points = norm.ppf(qmc_points)
                dW[:, :, i] = qmc_points

        # dW = qmc_points.reshape(N, m, self.dim) * np.sqrt(dt)  # convert QMC points into Brownian increments
        W = np.cumsum(dW, axis=1)  # standard Brownian motion

        if self.dim > 1:
            L = np.linalg.cholesky(self.sigma).T
            W = np.einsum('nmi,ij->nmj', W, L)

        t = np.linspace(dt, T, m)  # time grid
        S = np.zeros((N, m + 1, self.dim))  # initialize S
        S[:, 0, :] = S0  # set initial asset prices

        # apply geometric Brownian motion formula for remaining times
        S[:, 1:, :] = S0[None, None, :] * np.exp(
            (self.mu[None, None, :] - 0.5 * self.vol[None, None, :] ** 2) * t[None, :, None] + self.vol[None, None,
                                                                                               :] * W)

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
