import numpy as np
from scipy.stats import poisson

def esperanza_analitica(problema, i, xi):
    lam = problema.demanda_media[i]
    falta  = lam * poisson.sf(xi - 1, lam) - xi * poisson.sf(xi, lam)
    exceso = falta + xi - lam            # identidad (x-D)^+ = (D-x)^+ + x - lam
    return problema.gamma[i] * falta + problema.beta[i] * exceso

def esperanza_montecarlo(problema, i, xi, N=20000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    d = rng.poisson(problema.demanda_media[i], size=N)
    coste = problema.gamma[i]*np.maximum(d-xi,0) + problema.beta[i]*np.maximum(xi-d,0)
    return float(coste.mean())