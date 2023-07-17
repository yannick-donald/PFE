from numpy.linalg.linalg import inv
import numpy as np
from PricingMethod import PricingMethod

from Option import MaxCallOption
from Option import Option
from StochasticProcess import StochasticProcess, BlackScholesProcess
from RegressionBasis import RegressionBasis, ChebyshevBasis


class LongstaffSchwartz(PricingMethod):
    def __init__(self, option, basis: RegressionBasis, N):
        self.option = option
        self.basis = basis
        self.N = N

    def price(self, option: Option, process: StochasticProcess, m) -> float:
        # S0: Union[float, np.ndarray], N: int, m: int, T: float)
        # 1. Generate paths
        S = process.generate_paths(self.option.S0, self.N, m, self.option.T)
        m = S.shape[1] - 1  # number of time steps
        d = S.shape[2]  # number of dimensions

        # 2. Compute the payoff at the last time step
        V = self.option.payoff(S[:, m])

        # 3. Working backwards in time
        for t in range(m - 1, 0, -1):
            # Select in-the-money paths
            in_the_money = self.option.payoff(S[:, t]) > 0
            S_in_the_money = S[in_the_money, t]
            V_in_the_money = V[in_the_money]

            # Compute X and Y for the least squares problem
            H = self.basis.evaluate(S_in_the_money)
            X = H
            Y = V_in_the_money

            # Solve the least squares problem
            a_star = inv(X.T @ X) @ X.T @ Y

            # Compute the continuation values
            C_star = X @ a_star

            # Update option values where exercise is beneficial
            V[in_the_money] = np.where(
                self.option.payoff(S_in_the_money) > C_star,
                self.option.payoff(S_in_the_money),
                V_in_the_money)

        # 4. Compute and return the discounted expected payoff
        return np.exp(-process.r * self.option.T) * np.mean(V)


if __name__ == '__main__':
    r = np.array([0.05, 0.05, 0.05])
    S0 = np.array([100, 100, 100])
    mu = np.array([0.05, 0.05, 0.05])
    sigma = np.array([[0.04, 0.01, 0.01], [0.01, 0.04, 0.01], [0.01, 0.01, 0.04]])  # Covariance matrix
    K = 100
    T = 1
    N = 50000
    m = 50
    degree = 2

    # Objects
    option = MaxCallOption(K, T, S0)

    process = BlackScholesProcess(len(S0), r, sigma, mu)
    basis = ChebyshevBasis(degree)
    ls = LongstaffSchwartz(option, basis, N)

    price = ls.price(option, process, m)
    print('The max-call option price is', price)
