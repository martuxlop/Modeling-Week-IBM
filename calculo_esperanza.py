import numpy as np
from scipy.stats import poisson, nbinom

#funcion auxiliar para saber la distribucion de demanda
def obtener_distribucion(problema, i, q=0.999):

    params = problema.parametros_demanda[i]

    if problema.tipo_demanda == "poisson":

        mu = params["mu"]
        d_max = int(poisson.ppf(q, mu))

        demandas = np.arange(d_max + 1)
        probs = poisson.pmf(demandas, mu)

    elif problema.tipo_demanda == "binomial_negativa":

        r = params["r"]
        p = params["p"]

        d_max = int(nbinom.ppf(q, r, p))

        demandas = np.arange(d_max + 1)
        probs = nbinom.pmf(demandas, r, p)

    else:
        raise ValueError("Distribución de demanda no reconocida.")

    probs = probs / probs.sum()

    return demandas, probs

def esperanza_analitica_1(problema, i, xi):
    demandas, probs = obtener_distribucion(problema, i)
    costes = np.abs(demandas - xi)
    return float(np.sum(probs * costes))


def esperanza_montecarlo_1(problema, i, xi, N=20000, q=0.999):
    demandas, probs = obtener_distribucion(problema, i, q)

    muestras = np.random.choice(
        demandas,
        size=N,
        p=probs
    )
    return np.mean(np.abs(muestras - xi))

def esperanza_analitica_2(problema, i, xi):
    demandas, probs = obtener_distribucion(problema, i)
    costes = (
        problema.gamma[i] * np.maximum(demandas - xi, 0)
        + problema.beta[i] * np.maximum(xi - demandas, 0)
    )

    return float(np.sum(probs * costes))

def esperanza_montecarlo_2(problema, i, xi, N=20000):

    demandas, probs = obtener_distribucion(problema, i)

    muestras = np.random.choice(
        demandas,
        size=N,
        p=probs
    )

    costes = (
        problema.gamma[i] * np.maximum(muestras - xi, 0)
        + problema.beta[i] * np.maximum(xi - muestras, 0)
    )

    return float(np.mean(costes))