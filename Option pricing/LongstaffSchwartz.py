from numpy.linalg.linalg import inv
import numpy as np
from PricingMethod import PricingMethod

from Option import MaxCallOption
from Option import Option
from StochasticProcess import StochasticProcess, BlackScholesProcess
from RegressionBasis import RegressionBasis, ChebyshevBasis, LaguerreBasis,PolynomialBasis,HermiteBasis,PolynomialHortoBasis3D


class LongstaffSchwartz(PricingMethod):
    def __init__(self, option, basis: RegressionBasis, N):
        self.option = option
        self.basis = basis
        self.N = N

    def price(self, option: Option, process: StochasticProcess, m, qmc=None) -> float:
        # S0: Union[float, np.ndarray], N: int, m: int, T: float)
        # 1. Generate paths
        if qmc:
            S = process.generate_paths_qmc(self.option.S0, self.N, m, self.option.T, type_qmc=qmc)

        else:

            S = process.generate_paths(self.option.S0, self.N, m, self.option.T)

        m = S.shape[1] - 1  # number of time steps
        d = S.shape[2]  # number of dimensions

        # 2. Compute the payoff at the last time step
        V = self.option.payoff(S[:, m])

        # 3. Working backwards in time
        for t in range(m - 1, 3, -1):
            # Select in-the-money paths
            in_the_money = self.option.payoff(S[:, t]) > 0
            S_in_the_money = S[in_the_money, t]
            V_in_the_money = V[in_the_money]

            if self.basis.basis_name in ["ChebyshevBasis", "LaguerreBasis","PolynomialBasis","HermiteBasis"]:

                S_in_the_money_max = np.max(S_in_the_money, axis=1)  # on peut choisir  une autre vecteur

                # Compute X and Y for the least squares problem
                H = self.basis.evaluate(S_in_the_money_max)
            else:
                # Compute X and Y for the least squares problem
                H = self.basis.evaluate(S_in_the_money)

            X = H.T
            Y = V_in_the_money.reshape((len(V_in_the_money), 1))  # on peut choisir  une autre vecteur

            # Solve the least squares problem
            try:
                a_star = inv(X.T @ X) @ X.T @ Y
                # Compute the continuation values
                C_star = X @ a_star

                # Update option values where exercise is beneficial
                V[in_the_money] = np.where(
                    np.diag(self.option.payoff(S_in_the_money) > C_star * np.exp(process.r[0] * t)),
                    self.option.payoff(S_in_the_money),
                    V_in_the_money)

            # 4. Compute and return the discounted expected payoff
            except:
                pass

        # 4. Compute and return the discounted expected payoff
        return np.mean(V), S, X, t, S_in_the_money


if __name__ == '__main__':
    r = np.array([0.05, 0.05, 0.05])
    S0 = np.array([100, 100, 100])
    mu = np.array([0.05, 0.05, 0.05])
    sigma = np.array([[0.04, 0.01, 0.01], [0.01, 0.04, 0.01], [0.01, 0.01, 0.04]])  # Covariance matrix
    K = 100
    T = 1
    N = 1000
    m = 50
    degree = 2

    # Objects
    option = MaxCallOption(K, T, S0)

    process = BlackScholesProcess(len(S0), r, sigma, mu)
    basis = ChebyshevBasis(degree)
    ls = LongstaffSchwartz(option, basis, N)

    price = ls.price(option, process, m)
    print('The max-call option price is', price)
