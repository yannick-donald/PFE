from numpy.linalg.linalg import inv
import numpy as np
from PricingMethod import PricingMethod

from Option import MaxCallOption
from Option import *
from StochasticProcess import StochasticProcess, BlackScholesProcess
from RegressionBasis import RegressionBasis, ChebyshevBasis, LaguerreBasis, PolynomialBasis, HermiteBasis, \
    PolynomialHortoBasis3D
import time


class LongstaffSchwartz(PricingMethod):
    def __init__(self, option, basis: RegressionBasis, N):
        self.option = option
        self.basis = basis
        self.N = N

    def price(self, option: Option, process: StochasticProcess, m, qmc=None) -> float:
        # S0: Union[float, np.ndarray], N: int, m: int, T: float)
        # 1. Générer les chemins
        if qmc:
            S = process.generate_paths_qmc(self.option.S0, self.N, m, self.option.T, type_qmc=qmc)

        else:

            S = process.generate_paths(self.option.S0, self.N, m, self.option.T)

        m = S.shape[1] - 1  # number of time steps
        d = S.shape[2]  # number of dimensions

        # 2. Calculer le payoff à la dernière étape temporelle
        V = self.option.payoff(S[:, m])

        # 3. Travailler en arrière dans le temps
        for t in range(m - 1, 0, -1):
            # Select in-the-money paths
            in_the_money = self.option.payoff(S[:, t]) > 0
            in_the_money = in_the_money.reshape(self.N, )
            # print(in_the_money)
            # print(S)
            # return in_the_money, S,V

            S_in_the_money = S[in_the_money, t]
            V_in_the_money = V[in_the_money]

            if self.basis.basis_name in ["ChebyshevBasis", "LaguerreBasis", "PolynomialBasis", "HermiteBasis"]:

                S_in_the_money_max = np.max(S_in_the_money, axis=1)  # on peut choisir  une autre vecteur

                # Compute X and Y for the least squares problem
                H = self.basis.evaluate(S_in_the_money_max)
            elif self.basis.basis_name in ["MonomialBasis3D", "PolynomialHortoBasis3D", "PolynomialBasis3D"]:
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
        return np.mean(V)
        # return np.mean(V), S, X, t, S_in_the_money


if __name__ == '__main__':

    def test_max_option():
        # Paramètres pour le test
        dim = 1  # Nombre de dimensions (nombre d'actifs)
        r = np.array([0.05, ])  # Taux sans risque pour chaque actif
        vol = np.array([0.1, ])  # Volatilité pour chaque actif
        corr = np.array([1.0])  # Matrice de corrélation
        mu = np.array([0.06])  # Dérive pour chaque actif
        process = BlackScholesProcess(dim=dim, r=r, vol=vol, corr=corr, mu=mu)  # Processus Black-Scholes

        # Création de l'option MaxPut
        S0 = np.array([100])
        K = 100
        T = 1
        option = CallOption(K, T, S0)

        # Paramètres pour le test
        N = 10  # Nombre de trajectoires
        m = 50  # Nombre de pas de temps
        degree_values = [i for i in range(0, 3, 3)]  # Différentes valeurs de degré

        for degree in degree_values:
            # Créer la base de régression polynomiale avec le degré spécifié
            basis = PolynomialBasis(degree)
            ls = LongstaffSchwartz(option, basis, N)

            # Mesure du temps de calcul
            start_time = time.time()
            price, _, _, _, _ = ls.price(option, process, m)
            end_time = time.time()

            print(f'Le prix de l\'option MaxPut avec la base de régression polynomiale '
                  f'de degré {degree} pour N={N} est : {price:.4f}')
            print(f'Temps de calcul : {end_time - start_time:.6f} secondes\n')


    if __name__ == '__main__':
        # Tester l'option MaxPut avec différentes valeurs de degré de la base de régression
        test_max_option()
