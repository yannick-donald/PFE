import matplotlib.pyplot as plt
import numpy as np
from StochasticProcess import BlackScholesProcess

# Paramètres du processus
dim = 2
r = 0.05
sigma = np.array([0.2, 0.3])
mu = 0.1
rho = np.array([[1.0, 0.5], [0.5, 1.0]])

# Créer le processus de Black-Scholes
process = BlackScholesProcess(dim=dim, r=r, sigma=sigma, mu=mu, rho=rho)

# Paramètres de la simulation
S0 = 100
N = 1000
m = 100
T = 1.0

# Générer des chemins
paths = process.generate_paths(S0, N, m, T)

# Tracer quelques chemins pour chaque actif
plt.figure(figsize=(12, 6))

for i in range(dim):
    plt.subplot(dim, 1, i+1)
    for j in range(5):
        plt.plot(paths[j, :, i])
    plt.title(f'Asset {i+1}')
plt.tight_layout()
plt.show()
