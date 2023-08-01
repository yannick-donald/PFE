from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Optional
from scipy.special import eval_chebyt, eval_laguerre, eval_hermite
from itertools import *


class RegressionBasis(ABC):
    @abstractmethod
    def evaluate(self, x: Union[float, np.ndarray]) -> np.ndarray:
        pass


class ChebyshevBasis(RegressionBasis):
    def __init__(self, degree):
        self.degree = degree
        self.basis_name = "ChebyshevBasis"

    def evaluate(self, x):
        return eval_chebyt(np.arange(self.degree + 1)[:, None], x)


class LaguerreBasis(RegressionBasis):
    def __init__(self, degree):
        self.degree = degree
        self.basis_name = "LaguerreBasis"

    def evaluate(self, x):
        return eval_laguerre(np.arange(self.degree + 1)[:, None], x)


class HermiteBasis(RegressionBasis):
    def __init__(self, degree):
        self.degree = degree
        self.basis_name = "HermiteBasis"

    def evaluate(self, x):
        return eval_hermite(np.arange(self.degree + 1)[:, None], x)


class PolynomialBasis(RegressionBasis):
    def __init__(self, degree):
        self.degree = degree
        self.basis_name = "PolynomialBasis"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # Create an array of powers to raise x to
        powers = np.arange(self.degree + 1).reshape(-1, 1, 1)

        # Raise x to these powers
        polynomial_basis_values = x ** powers

        return polynomial_basis_values


class PolynomialBasis3D(RegressionBasis):
    def __init__(self, degree: int):
        self.degree = degree
        self.basis_name = "PolynomialBasis3D"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        powers = np.arange(1, self.degree + 1)

        polynomial_basis_values = np.zeros((x.shape[0], x.shape[1] * self.degree + 1))
        polynomial_basis_values[:, 0] = np.ones(x.shape[0])

        N_actif = x.shape[1]
        p = x[:, 0].reshape(-1, 1)

        polynomial_basis_values[:, 1:(self.degree + 1)] = p ** powers
        debut = self.degree + 1
        for i in range(1, N_actif):
            p = x[:, i].reshape(-1, 1)

            fin = debut + self.degree

            polynomial_basis_values[:, debut:fin] = p ** powers

            debut = fin

            fin = debut + self.degree

        return polynomial_basis_values


class MonomialBasis3D(RegressionBasis):
    def __init__(self, degree: int):
        self.degree = degree
        self.basis_name = "MonomialBasis3D"


    def evaluate(self, x: np.ndarray) -> np.ndarray:
        powers = np.arange(1, self.degree + 1)

        monomial_basis_values = np.zeros((x.shape[0], x.shape[1] * self.degree + 1))
        monomial_basis_values[:, 0] = np.ones(x.shape[0])

        N_actif = x.shape[1]
        p = x[:, 0].reshape(-1, 1)

        monomial_basis_values[:, 1:(self.degree + 1)] = p ** powers
        debut = self.degree + 1
        for i in range(1, N_actif):
            p = x[:, i].reshape(-1, 1)

            fin = debut + self.degree

            monomial_basis_values[:, debut:fin] = p ** powers

            debut = fin

            fin = debut + self.degree

        return monomial_basis_values


class PolynomialHortoBasis3D(RegressionBasis):
    def __init__(self, degree: int, basis: RegressionBasis):
        self.degree = degree
        self.horto_basis = basis
        self.basis_name = "PolynomialHortoBasis3D"

    def P(self, s, k, n):
        B = product(list(range(n)), repeat=k)
        p = []
        H = self.horto_basis.evaluate(s)
        # return H

        for i in B:

            if sum(i) < n and sum(i) != 0:
                # print(i)

                init = 1

                for j in range(k):
                    init *= H[i[j], j]

                p.append(init)
        return p

    def evaluate(self, x: Union[float, np.ndarray]) -> np.ndarray:
        M_p = []
        k = len(x[0])
        n = self.degree + 1

        for Nmc in range(x.shape[0]):
            M_p.append(self.P(x[Nmc], k, n))

        return np.array(M_p)

    class PolynomialBasisPayOff(RegressionBasis):

        def __init__(self, degree: int, basis: RegressionBasis, pay_off):
            self.degree = degree
            self.horto_basis = basis
            self.pay_off = pay_off

        def evaluate(self, x: np.ndarray) -> np.ndarray:
            powers = np.arange(1, self.degree + 1)

            polynomial_basis_values = np.zeros((x.shape[0], x.shape[1] * self.degree + 1))
            polynomial_basis_values[:, 0] = np.ones(x.shape[0])

            N_actif = x.shape[1]
            p = x[:, 0].reshape(-1, 1)

            polynomial_basis_values[:, 1:(self.degree + 1)] = p ** powers
            debut = self.degree + 1
            for i in range(1, N_actif):
                p = x[:, i].reshape(-1, 1)

                fin = debut + self.degree

                polynomial_basis_values[:, debut:fin] = p ** powers

                debut = fin

                fin = debut + self.degree

            polynomial_basis_values

            return polynomial_basis_values
