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

    def _pmf(self, i, valores):
        """Evalúa la pmf de la demanda de la zona i en `valores` (array de enteros)."""
        params = self.parametros_demanda[i]

        if self.tipo_demanda == "poisson":
            return poisson.pmf(valores, params["mu"])

        elif self.tipo_demanda == "binomial_negativa":
            return nbinom.pmf(valores, params["r"], params["p"])

        else:
            raise ValueError("Distribución de demanda no reconocida.")

    def media_varianza(self, i):
        """Media y varianza teóricas de la demanda de la zona i."""
        params = self.parametros_demanda[i]

        if self.tipo_demanda == "poisson":
            mu = float(params["mu"])
            return mu, mu

        elif self.tipo_demanda == "binomial_negativa":
            r, p = params["r"], params["p"]
            media = r * (1 - p) / p
            return float(media), float(media / p)

        else:
            raise ValueError("Distribución de demanda no reconocida.")

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
            d_max = int(poisson.ppf(q, params["mu"]))

        elif self.tipo_demanda == "binomial_negativa":
            d_max = int(nbinom.ppf(q, params["r"], params["p"]))

        else:
            raise ValueError("Distribución de demanda no reconocida.")

        demandas = np.arange(d_max + 1)
        probs = self._pmf(i, demandas)
        probs = probs / probs.sum()

        self._cache_distribucion[clave] = (demandas, probs)
        return demandas, probs

    def distribucion_ventana(self, i, k_izq=3.0, k_der=4.0, masa_minima=0.999):
        """
        Discretización de la demanda de la zona i en una ventana centrada en
        la media:

            [ max(0, media - k_izq * sigma) ,  media + k_der * sigma ]

        con soporte de enteros contiguos (paso 1) y probabilidades
        renormalizadas. Pensada para el módulo de Amplitude Estimation: el
        tamaño de la ventana K determina el número de qubits del registro
        índice (n = ceil(log2(K))).

        La ventana es asimétrica por defecto (k_der > k_izq) porque Poisson
        y, sobre todo, la Binomial Negativa sobredispersa tienen cola derecha
        pesada, y la función de coste crece en esa cola: cortarla introduce
        más sesgo que cortar la izquierda.

        Si la masa capturada por la ventana es menor que `masa_minima`,
        lanza ValueError (el sesgo de truncamiento podría dominar el error
        del estimador): basta ampliar k_der / k_izq.

        Devuelve (demandas, probs), con demandas = d_lo, d_lo+1, ..., d_hi.
        """
        clave = ("ventana", i, k_izq, k_der)
        if clave in self._cache_distribucion:
            return self._cache_distribucion[clave]

        media, varianza = self.media_varianza(i)
        sigma = np.sqrt(varianza)

        d_lo = max(0, int(np.floor(media - k_izq * sigma)))
        d_hi = int(np.ceil(media + k_der * sigma))

        demandas = np.arange(d_lo, d_hi + 1)
        probs = self._pmf(i, demandas)

        masa = probs.sum()
        if masa < masa_minima:
            raise ValueError(
                f"La ventana [{d_lo}, {d_hi}] de la zona {i} captura masa "
                f"{masa:.6f} < {masa_minima}. Amplíe k_izq/k_der."
            )

        probs = probs / masa

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
