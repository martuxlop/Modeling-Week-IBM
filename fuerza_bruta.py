# fuerza_bruta.py

import numpy as np

from calculo_esperanza import coste_total


def _asignaciones_factibles(n, C):
    """
    Genera todas las asignaciones x = (x_1, ..., x_n) con x_i entero >= 0 y
    sum_i x_i <= C. El número de asignaciones es C(C + n, n), por lo que solo
    es viable para instancias pequeñas (n y C reducidos).
    """
    if n == 1:
        for v in range(C + 1):
            yield (v,)
        return

    for v in range(C + 1):
        for resto in _asignaciones_factibles(n - 1, C - v):
            yield (v,) + resto


def asignar_fuerza_bruta(problema, esperanza_zona):
    """
    Resuelve el problema por enumeración exhaustiva: evalúa la función
    objetivo en TODAS las asignaciones factibles y devuelve la de menor coste.

    Es el óptimo global de referencia para validar empíricamente el greedy.
    Pensado para instancias pequeñas (el espacio de búsqueda crece de forma
    combinatoria). Conviene usarlo con la esperanza analítica para tener una
    referencia exacta y sin ruido.
    """
    mejor_x = None
    mejor_coste = np.inf

    for x in _asignaciones_factibles(problema.n, problema.C):
        x = np.array(x, dtype=int)
        c = coste_total(problema, x, esperanza_zona)
        if c < mejor_coste:
            mejor_coste = c
            mejor_x = x

    return mejor_x, float(mejor_coste)
