import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np

class Graphics(ABC):
    @abstractmethod
    def plot_price_path(self, option, path):
        pass

    @abstractmethod
    def plot_payoff(self, option):
        pass


class EuropeanOptionGraphics(Graphics):
    def plot_price_path(self, option, path):
        plt.figure()
        plt.plot(path)
        plt.title("Simulated Paths for European Option")
        plt.show()

    def plot_payoff(self, option):
        plt.figure()
        plt.plot([option.payoff(spot) for spot in np.linspace(0, 200, 400)])
        plt.title("Payoff for European Option")
        plt.show()


class AmericanOptionGraphics(Graphics):
    def plot_price_path(self, option, path):
        plt.figure()
        plt.plot(path)
        plt.title("Simulated Paths for American Option")
        plt.show()

    def plot_payoff(self, option):
        plt.figure()
        plt.plot([option.payoff(spot) for spot in np.linspace(0, 200, 400)])
        plt.title("Payoff for American Option")
        plt.show()


class AsianOptionGraphics(Graphics):
    def plot_price_path(self, option, path):
        plt.figure()
        plt.plot(path)
        plt.title("Simulated Paths for Asian Option")
        plt.show()

    def plot_payoff(self, option):
        plt.figure()
        plt.plot([option.payoff(spot) for spot in np.linspace(0, 200, 400)])
        plt.title("Payoff for Asian Option")
        plt.show()
