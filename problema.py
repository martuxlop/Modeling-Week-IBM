# problema.py

import numpy as np
from scipy.stats import poisson


class Problema:
    """
    Clase base.

    Define la estructura común:
        - restricciones / factibilidad,
        - distribución de demanda Poisson por zona,
        - muestreo de la demanda.

    El coste determinista de cada zona se define en las subclases mediante
    el método `coste(i, d, xi)`.
    """

    def __init__(self, C, parametros_demanda):
        self.C = int(C)
        self.parametros_demanda = parametros_demanda
        self.n = len(parametros_demanda)

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
        Devuelve (demandas, probs) de la zona i, truncada al cuantil q
        y renormalizada.

        Se asume siempre:

            d_i ~ Poisson(mu_i)
        """
        clave = (i, q)

        if clave in self._cache_distribucion:
            return self._cache_distribucion[clave]

        mu = self.parametros_demanda[i]["mu"]

        d_max = int(poisson.ppf(q, mu))
        demandas = np.arange(d_max + 1)
        probs = poisson.pmf(demandas, mu)

        probs = probs / probs.sum()

        self._cache_distribucion[clave] = (demandas, probs)

        return demandas, probs

    def muestrear(self, i, N, rng):
        """
        Muestrea N realizaciones de la demanda real de la zona i.

        No trunca la distribución: es Monte Carlo sobre la Poisson original.
        """
        mu = self.parametros_demanda[i]["mu"]

        return rng.poisson(mu, size=N)

    def coste(self, i, d, xi):
        raise NotImplementedError


class ProblemaPenalizado(Problema):
    """
    Problema 2:

        min (1/n) E[ sum_i gamma_i max(d_i - x_i, 0)
                   + beta_i max(x_i - d_i, 0) ]

        s.a.
            sum_i x_i <= C
            x_i entero no negativo
    """

    def __init__(self, C, parametros_demanda, gamma, beta):
        super().__init__(C, parametros_demanda)

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

    Es el caso particular del problema penalizado con gamma_i = beta_i = 1.
    """

    def __init__(self, C, parametros_demanda):
        n = len(parametros_demanda)

        super().__init__(
            C=C,
            parametros_demanda=parametros_demanda,
            gamma=np.ones(n),
            beta=np.ones(n),
        )