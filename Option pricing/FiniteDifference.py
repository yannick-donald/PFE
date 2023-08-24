from PricingMethod import *


class FiniteDifference(PricingMethod):
    def __init__(self, option: Option):
        self.option = Option
        self.N = 99
        self.M = 4999
        self.L = 20
        self.V_ame = np.zeros((self.M + 2, self.N + 2))
        self.S = np.linspace(0, self.L, self.N + 2)

    def conditionfinale(self, S):
        return self.option.payoff(S)

    def price(self, option: Option, process: StochasticProcess) -> float:

        for i in range(self.N + 2):
            self.V_ame[self.M + 1][i] = self.conditionfinale(self.S[i])

        for n in range(self.M + 1, 0, -1):
            for i in range(0, self.N + 1):
                self.V_ame[n - 1][i] = max(K - S[i], V_ame[n][i] + dt * (
                        r * S[i] * (V_ame[n][i + 1] - V_ame[n][i - 1]) / (2 * ds) + 1 / 2 * sigma ** 2 * S[
                    i] ** 2 * (V_ame[n][i + 1] + V_ame[n][i - 1] - 2 * V_ame[n][i]) / (ds * ds) - r * V_ame[n][i]))

            V_eur[n - 1][N + 1] = V_eur[n - 1][N]  # condition de neuman

            V_eur[n - 1][0] = V_eur[n - 1][1] + ds  # condtion de neuman

            V_ame[n - 1][N + 1] = V_ame[n - 1][N]  # condition de neuman

            V_ame[n - 1][0] = V_ame[n - 1][1] + ds  #

        pass
