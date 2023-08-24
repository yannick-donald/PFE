import numpy as np
import matplotlib.pyplot as plt
import time


class Binomial():
# ----------------------- Initialization of the attributes -------------------------------------------------------------
    def __init__(self, r, T, S0, K, sigma, N):
        self.r = r
        self.T = T
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.num_steps = N
        self.dt = T / N
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.stock_prices = np.zeros((N + 1, N + 1))
        self.option_values = np.zeros((N + 1, N + 1))
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------- Generation of the nodes ----------------------------------------------------------------------
    def compute_stock_price(self):
        for i in range(self.num_steps + 1):
            for j in range(i + 1):
                self.stock_prices[j, i] = self.S0 * (self.u ** j) * (self.d ** (i - j))
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------- Backward Iteration -------------------------------------------------------------
    def compute_option_values(self):
        self.option_values[:, self.num_steps] = np.maximum(self.stock_prices[:, self.num_steps] - self.K, 0)

        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                expected_value = np.exp(-self.r * self.dt) * (
                            self.p * self.option_values[j, i + 1] + (1 - self.p) * self.option_values[j + 1, i + 1])
                immediate_value = np.maximum(self.stock_prices[j, i] - self.K, 0)
                self.option_values[j, i] = np.maximum(expected_value, immediate_value)
# ----------------------- Initialization of the attributes -------------------------------------------------------------
    def compute_option_price(self):
        return self.option_values[0, 0]
# ----------------------- Initialization of the attributes -------------------------------------------------------------
if __name__ == '__main__':

    # Create an instance of BinomialOptionPricing
    binomial_option = Binomial(r=0.05, T=1.0, S0=100, K=100, sigma=0.2, N=100)

    # Calculate stock prices and option values
    binomial_option.compute_stock_price()
    binomial_option.compute_option_values()

    # Calculate and print option price
    option_price = binomial_option.compute_option_price()
    print("Option Price:", option_price)

    # Visualize stock price path:
    stock_price = binomial_option.compute_stock_price()

    plt.figure(figsize=(10, 6))
    t = np.linspace(0,1,101)
    for i in range(2):
        plt.scatter(t,binomial_option.stock_prices[ :,i])
    plt.title("Evolution of Underlying Stock Price")
    plt.xlabel("Number of Steps")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()
    plt.show()

    # Visualize stock price path:
    """plt.figure(figsize=(10, 6))
    for i in range(binomial_option.num_steps + 1):
        plt.plot(binomial_option.option_values[:, i], label=f"Step {i}")
    plt.title("Evolution of Option Price")
    plt.xlabel("Number of Steps")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid()
    plt.show()"""