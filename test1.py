import numpy as np
from numpy.linalg import inv, cholesky
from scipy.special import eval_chebyt


class MaxCallOption:
    def __init__(self, K, T):
        self.K = K
        self.T = T

    def payoff(self, S):
        return np.maximum(np.max(S, axis=2) - self.K, 0)


class BlackScholesProcess:
    def __init__(self, dim, r, sigma, mu):
        self.dim = dim
        self.r = r
        self.sigma = sigma
        self.mu = mu

    def generate_paths(self, S0, N, m, T):
        dt = T / m
        dW = np.random.normal(size=(N, m, self.dim)) * np.sqrt(dt)  # increments of standard Brownian motion
        W = np.cumsum(dW, axis=1)  # standard Brownian motion
        t = np.linspace(dt, T, m)  # time grid
        if self.dim > 1:
            L = np.linalg.cholesky(
                self.sigma).T  # Cholesky decomposition of the covariance matrix, transposed to get lower triangular matrix
            W = np.array([np.dot(L, w.T).T for w in W])  # correlated Brownian motion
        S = np.zeros((N, m + 1, self.dim))  # initialize S
        S[:, 0, :] = S0  # set initial asset prices
        # apply geometric Brownian motion formula for remaining times
        S[:, 1:, :] = S0[:, None] * np.exp(((self.mu - 0.5 * np.diag(self.sigma))[:, None] * t + W.T).T)
        return S


class ChebyshevBasis:
    def __init__(self, degree):
        self.degree = degree

    def evaluate(self, x):
        return eval_chebyt(np.arange(self.degree + 1)[:, None], x)


class LongstaffSchwartz:
    def __init__(self, option, basis, N):
        self.option = option
        self.basis = basis
        self.N = N

    def price(self, process, S0, m):
        # 1. Generate paths
        S = process.generate_paths(S0, self.N, m, self.option.T)
        m = S.shape[1] - 1  # number of time steps
        d = S.shape[2]  # number of dimensions

        # 2. Compute the payoff at the last time step
        #V = self.option.payoff(S[:, m])
        V = self.option.payoff(S[:, m, :])


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
                V[in_the_money])

        # 4. Compute and return the discounted expected payoff
        return np.exp(-process.r * self.option.T) * np.mean(V)


if __name__ == '__main__':
    # Parameters
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
    option = MaxCallOption(K, T)
    process = BlackScholesProcess(len(S0), r, sigma, mu)
    basis = ChebyshevBasis(degree)
    ls = LongstaffSchwartz(option, basis, N)

    # Price
    price = ls.price(process, S0, m)
    print('The max-call option price is', price)
