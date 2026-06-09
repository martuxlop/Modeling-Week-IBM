import numpy as np
from scipy.stats import poisson

def esperanza_analitica_1(problema, i, xi, q=0.999):
    lam = problema.demanda_media[i]
    d_max = int(poisson.ppf(q, lam))    # valor max de la demanda para cubrir el 99.9% de los casos
    demandas = np.arange(d_max + 1)
    probs = poisson.pmf(demandas, lam)
    probs = probs / probs.sum()   # normalización
    costes = np.abs(demandas - xi)
    return np.sum(probs * costes)


def esperanza_montecarlo_1(problema, i, xi, N=20000, q=0.999):
    lam = problema.demanda_media[i]
    d_max = int(poisson.ppf(q, lam))
    demandas = np.arange(d_max + 1)
    probs = poisson.pmf(demandas, lam)
    probs = probs / probs.sum()
    
    muestras = np.random.choice(
        demandas,
        size=N,
        p=probs
    )
    return np.mean(np.abs(muestras - xi))

def esperanza_analitica_2(problema, i, xi, q=0.999):
    lam = problema.demanda_media[i]
    d_max = int(poisson.ppf(q, lam))
    demandas = np.arange(d_max + 1)
    probs = poisson.pmf(demandas, lam)
    probs = probs / probs.sum()
    costes = (
        problema.gamma[i] * np.maximum(demandas - xi, 0)
        + problema.beta[i] * np.maximum(xi - demandas, 0)
    )
    return np.sum(probs * costes)

def esperanza_montecarlo_2(problema, i, xi, N=20000, q=0.999, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    lam = problema.demanda_media[i]
    d_max = int(poisson.ppf(q, lam))
    demandas = np.arange(d_max + 1)
    probs = poisson.pmf(demandas, lam)
    probs = probs / probs.sum()

    muestras = rng.choice(
        demandas,
        size=N,
        p=probs
    )

    costes = (
        problema.gamma[i] * np.maximum(muestras - xi, 0)
        + problema.beta[i] * np.maximum(xi - muestras, 0)
    )

    return float(np.mean(costes))