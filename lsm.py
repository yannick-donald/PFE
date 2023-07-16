import numpy as np
from scipy.special import laguerre, chebyt
from scipy.stats import norm


class MaxCallOptionPricer:
    def __init__(self, N, m, T, basis_functions):
        self.N = N
        self.m = m
        self.T = T
        self.basis_functions = basis_functions
        self.optimal_coefficients = None

    def generate_bs_stock_paths(self, S0, r, sigma):
        paths = np.zeros((self.N, self.m + 1, len(S0)))
        dt = self.T / self.m

        for i in range(self.N):
            for j in range(len(S0)):
                paths[i, 0, j] = S0[j]  # Set initial stock prices
                for k in range(self.m):
                    z = np.random.randn(len(S0))
                    paths[i, k + 1, j] = paths[i, k, j] * np.exp((r[j] - 0.5 * sigma[j] ** 2) * dt +
                                                                 sigma[j] * np.sqrt(dt) * z[j])

        return paths

    def compute_option_price(self, S0, r, sigma):
        paths = self.generate_bs_stock_paths(S0, r, sigma)

        # Step 2: Set terminal values
        terminal_values = np.amax(paths[:, -1, :], axis=1)

        # Step 3: Backward calculation
        for i in range(self.m - 1, 0, -1):
            current_paths = paths[:, i, :]
            continuation_values = np.zeros_like(current_paths)
            exercise_values = np.zeros_like(current_paths)

            for j in range(self.N):
                if np.amax(current_paths[j, :]) > 0:  # Check if option is in the money
                    continuation_values[j] = self.compute_continuation_value(current_paths[j, :])
                    exercise_values[j] = np.amax(current_paths[j, :])

            # Update option values
            terminal_values = np.where(exercise_values >= continuation_values, exercise_values, terminal_values)

        return np.mean(terminal_values)

    def compute_continuation_value(self, S):
        X = np.vstack([basis_func(S) for basis_func in self.basis_functions]).T
        return np.dot(X, self.optimal_coefficients)

    def solve_regression(self, paths, values):
        X = np.vstack([basis_func(paths) for basis_func in self.basis_functions]).T
        self.optimal_coefficients = np.linalg.inv(X.T @ X) @ X.T @ values


def generate_bs_multidimensional_stock_paths(S0, r, sigma, T, N, m):
    paths = np.zeros((N, m + 1, len(S0)))
    dt = T / m

    for i in range(N):
        for j in range(len(S0)):
            paths[i, 0, j] = S0[j]  # Set initial stock prices
            for k in range(m):
                z = np.random.randn(len(S0))
                paths[i, k + 1, j] = paths[i, k, j] * np.exp((r[j] - 0.5 * sigma[j] ** 2) * dt +
                                                             sigma[j] * np.sqrt(dt) * z[j])

    return paths


# Usage example
N = 10000  # Number of paths
m = 100  # Number of exercise dates
T = 3.0  # Time to maturity
S0 = [100.0, 120.0, 90.0, 110.0, 95.0]  # Initial stock prices for 5 underlying assets
r = [0.05, 0.06, 0.04, 0.03, 0.05]  # Risk-free interest rates for the assets
sigma = [0.2, 0.25, 0.3, 0.35, 0.4]  # Volatility for the assets

# Define Laguerre and Chebyshev basis functions
laguerre_basis_functions = [lambda x: laguerre(0)(x), lambda x: laguerre(1)(x), lambda x: laguerre(2)(x)]
chebyshev_basis_functions = [lambda x: chebyt(0)(x), lambda x: chebyt(1)(x), lambda x: chebyt(2)(x)]
"""
option_pricer_laguerre = MaxCallOptionPricer(N, m, T, laguerre_basis_functions)
option_pricer_chebyshev = MaxCallOptionPricer(N, m, T, chebyshev_basis_functions)

option_price_laguerre = option_pricer_laguerre.compute_option_price(S0, r, sigma)
option_price_chebyshev = option_pricer_chebyshev.compute_option_price(S0, r, sigma)"""

# print("Max Call Option Price (Laguerre):", option_price_laguerre)
# print("Max Call Option Price (Chebyshev):", option_price_chebyshev)

if __name__ == '__main__':
    option_pricer_laguerre = MaxCallOptionPricer(N, m, T, laguerre_basis_functions)
    option_price_laguerre = option_pricer_laguerre.compute_option_price(S0, r, sigma)
