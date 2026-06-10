# calculo_esperanza.py

import numpy as np


def esperanza_analitica(problema, i, xi):
    """
    Esperanza exacta del coste de la zona i para la asignación xi, calculada
    sobre la distribución (truncada y cacheada) de la demanda.
    """
    demandas, probs = problema.distribucion(i)
    return float(np.sum(probs * problema.coste(i, demandas, xi)))


def coste_total(problema, x, esperanza_zona):
    """
    Valor de la función objetivo para una asignación x arbitraria:

        F(x) = (1/n) sum_i E[ coste_i(d_i, x_i) ]

    `esperanza_zona` es el método de cálculo de la esperanza por zona
    (analítico, Monte Carlo o AE), con la firma esperanza_zona(problema, i, xi).

    Es un evaluador independiente del algoritmo de optimización: sirve tanto
    para puntuar candidatas en la búsqueda por fuerza bruta como para verificar
    de forma autónoma el coste devuelto por el greedy.
    """
    return float(np.mean([
        esperanza_zona(problema, i, int(x[i])) for i in range(problema.n)
    ]))


def crear_esperanza_montecarlo(problema, rng, N=20000):
    """
    Construye un estimador Monte Carlo del coste por zona.

    Muestrea N demandas reales de cada zona UNA sola vez (con el generador
    reproducible `rng`) y reutiliza esas muestras para cualquier asignación
    xi. Esto evita remuestrear en cada llamada del greedy y, al usar números
    aleatorios comunes, mantiene coherentes las ganancias marginales.

    Devuelve una función con la firma esperanza_zona(problema, i, xi),
    compatible con `asignar_greedy`.
    """
    muestras = {i: problema.muestrear(i, N, rng) for i in range(problema.n)}

    def esperanza_montecarlo(problema, i, xi):
        return float(np.mean(problema.coste(i, muestras[i], xi)))

    return esperanza_montecarlo
