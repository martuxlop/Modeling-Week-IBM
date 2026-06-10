# problema.py

import numpy as np
from scipy.stats import poisson, nbinom


def parametros_binomial_negativa(media, varianza):
    """
    Convierte (media, varianza) en los parámetros (r, p) de la Binomial
    Negativa con la parametrización de scipy.stats.nbinom / numpy:

        media     = r (1 - p) / p
        varianza  = r (1 - p) / p**2 = media / p

    de donde:

        p = media / varianza
        r = media**2 / (varianza - media)

    Requiere sobredispersión (varianza > media). Esta es justamente la
    situación que modela la mixtura Gamma-Poisson (lambda ~ Gamma), que da
    lugar a la Binomial Negativa y a una varianza mayor que la media, a
    diferencia de la Poisson (donde varianza = media).

    Devuelve un dict {"r": r, "p": p} listo para usar como entrada de
    `parametros_demanda`.
    """
    if media <= 0:
        raise ValueError("La media debe ser positiva.")
    if varianza <= media:
        raise ValueError(
            "La Binomial Negativa requiere varianza > media (sobredispersión)."
        )

    p = media / varianza
    r = media ** 2 / (varianza - media)
    return {"r": r, "p": p}


class Problema:
    """
    Clase base.

    Define la estructura común a todos los problemas:
        - restricciones / factibilidad,
        - la distribución de demanda por zona (cacheada),
        - el muestreo de la variable aleatoria de demanda.

    El coste determinista de cada zona se define en las subclases mediante
    el método `coste(i, d, xi)`.
    """

    def __init__(self, C, tipo_demanda, parametros_demanda):
        self.C = int(C)
        self.tipo_demanda = tipo_demanda
        self.parametros_demanda = parametros_demanda
        self.n = len(parametros_demanda)

        # Cache de distribuciones truncadas por (zona, cuantil de truncamiento).
        self._cache_distribucion = {}

    def es_factible(self, x):
        x = np.atleast_1d(np.asarray(x))

        return (
            len(x) == self.n
            and np.all(x >= 0)
            and np.all(x.astype(int) == x)
            and np.sum(x) <= self.C
        )

    def distribucion(self, i, q=0.999):
        """
        Devuelve (demandas, probs) de la zona i, truncada al cuantil q y
        renormalizada. El resultado se cachea porque es constante por zona.
        """
        clave = (i, q)
        if clave in self._cache_distribucion:
            return self._cache_distribucion[clave]

        params = self.parametros_demanda[i]

        if self.tipo_demanda == "poisson":
            mu = params["mu"]
            d_max = int(poisson.ppf(q, mu))
            demandas = np.arange(d_max + 1)
            probs = poisson.pmf(demandas, mu)

        elif self.tipo_demanda == "binomial_negativa":
            r = params["r"]
            p = params["p"]
            d_max = int(nbinom.ppf(q, r, p))
            demandas = np.arange(d_max + 1)
            probs = nbinom.pmf(demandas, r, p)

        else:
            raise ValueError("Distribución de demanda no reconocida.")

        probs = probs / probs.sum()

        self._cache_distribucion[clave] = (demandas, probs)
        return demandas, probs

    def muestrear(self, i, N, rng):
        """
        Muestrea N realizaciones de la demanda real de la zona i usando el
        generador `rng` (np.random.Generator). No trunca la distribución:
        es Monte Carlo sobre la variable aleatoria original.
        """
        params = self.parametros_demanda[i]

        if self.tipo_demanda == "poisson":
            return rng.poisson(params["mu"], size=N)

        elif self.tipo_demanda == "binomial_negativa":
            return rng.negative_binomial(params["r"], params["p"], size=N)

        else:
            raise ValueError("Distribución de demanda no reconocida.")

    def coste(self, i, d, xi):
        """Coste determinista de la zona i dada la demanda d y la asignación xi."""
        raise NotImplementedError


class ProblemaPenalizado(Problema):
    """
    Problema 2:

        min (1/n) E[ sum_i gamma_i max(d_i - x_i, 0)
                   + beta_i max(x_i - d_i, 0) ]

        s.a.
            sum_i x_i <= C
            x_i entero no negativo

    gamma_i: coste de oportunidad (demanda no atendida).
    beta_i:  coste de conductor inutilizado.
    """

    def __init__(self, C, tipo_demanda, parametros_demanda, gamma, beta):
        super().__init__(C, tipo_demanda, parametros_demanda)

        self.gamma = np.array(gamma, dtype=float)
        self.beta = np.array(beta, dtype=float)

        if len(self.gamma) != self.n or len(self.beta) != self.n:
            raise ValueError("gamma y beta deben tener longitud n.")

    def coste(self, i, d, xi):
        d = np.asarray(d)
        return (
            self.gamma[i] * np.maximum(d - xi, 0)
            + self.beta[i] * np.maximum(xi - d, 0)
        )


class ProblemaAbsoluto(ProblemaPenalizado):
    """
    Problema 1:

        min (1/n) E[ sum_i |x_i - d_i| ]

        s.a.
            sum_i x_i <= C
            x_i entero no negativo

    Es el caso particular del problema penalizado con gamma_i = beta_i = 1,
    ya que |d - x| = max(d - x, 0) + max(x - d, 0).
    """

    def __init__(self, C, tipo_demanda, parametros_demanda):
        n = len(parametros_demanda)
        super().__init__(
            C,
            tipo_demanda,
            parametros_demanda,
            gamma=np.ones(n),
            beta=np.ones(n),
        )
