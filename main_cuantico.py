# main_cuantico.py
"""
Validación incremental y resolución del problema con Amplitude Estimation.

El script sigue la metodología de depuración cuántica del enunciado
("verify amplitudes with Statevector, compare with analytical calculations"):
valida cada pieza por separado ANTES de usar el algoritmo completo, de modo
que si algo falla se sabe exactamente en qué componente está el error.

    PASO 1: la carga de distribución P pone la pmf en las amplitudes.
    PASO 2: el operador A codifica la esperanza en P(objetivo = 1).
    PASO 3: esperanza_ae (IAE completo) coincide con el cálculo analítico.
    PASO 4: resolución del problema: greedy + AE = greedy + analítico.
"""

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from problema import ProblemaAbsoluto, ProblemaPenalizado
from greedy import asignar_greedy
from calculo_esperanza import esperanza_analitica
from calculo_esperanza_cuantico import (
    construir_operador_A,
    esperanza_ae,
    crear_esperanza_ae,
    C_RESCALADO,
)


def esperanza_ventana(problema, i, xi):
    """Esperanza exacta sobre la distribución de ventana (la referencia justa
    para validar AE, que trabaja con esa misma discretización)."""
    demandas, probs = problema.distribucion_ventana(i)
    return float(np.sum(probs * problema.coste(i, demandas, xi)))


def probabilidad_objetivo_1(circuito, n_indice):
    """P(qubit objetivo = 1) calculada exactamente con el statevector.
    El qubit objetivo es el n_indice (el siguiente al registro índice)."""
    sv = Statevector(circuito)
    probs = np.abs(sv.data) ** 2
    indices = np.arange(len(probs))
    mascara = (indices >> n_indice) & 1 == 1
    return float(probs[mascara].sum())


def paso_1_carga(problema, i):
    print("\n--- PASO 1: carga de distribucion P ---")
    demandas, probs = problema.distribucion_ventana(i)
    K = len(demandas)
    n = max(1, int(np.ceil(np.log2(K))))
    N = 2 ** n

    p = np.zeros(N)
    p[:K] = probs
    p = p / p.sum()

    from qiskit.circuit.library import StatePreparation
    qc = QuantumCircuit(n)
    qc.append(StatePreparation(np.sqrt(p)), range(n))

    # |amplitud_j|^2 debe reproducir la pmf discretizada.
    sv = Statevector(qc)
    p_medida = np.abs(sv.data) ** 2

    error = np.max(np.abs(p_medida - p))
    print(f"zona {i}: ventana [{demandas[0]}, {demandas[-1]}], K={K} valores, "
          f"n={n} qubits (padding hasta {N})")
    print(f"max |p_statevector - p_objetivo| = {error:.2e}")
    assert error < 1e-12, "La carga de distribucion no reproduce la pmf."
    print("OK: las amplitudes al cuadrado reproducen la pmf.")


def paso_2_operador_A(problema, i, xi):
    print("\n--- PASO 2: operador A (carga + coste) ---")
    demandas, probs = problema.distribucion_ventana(i)
    A, F, n, exacto = construir_operador_A(problema, i, xi, demandas, probs)
    assert exacto is None

    # (a) P(objetivo=1) exacta del circuito vs la formula teorica
    #     a = sum_j p_j sin^2(pi/4 + (pi c / 2)(f~_j - 1/2)).
    a_circuito = probabilidad_objetivo_1(A, n)

    d_lo = int(demandas[0])
    N = 2 ** n
    grid = np.arange(d_lo, d_lo + N)
    p_pad = np.zeros(N)
    p_pad[:len(probs)] = probs
    f_vals = np.asarray(problema.coste(i, grid, xi), dtype=float)
    f_norm = (f_vals - f_vals.min()) / (f_vals.max() - f_vals.min())
    theta = np.pi / 4 + (np.pi * C_RESCALADO / 2) * (f_norm - 0.5)
    a_teorica = float(np.sum(p_pad * np.sin(theta) ** 2))

    print(f"a (circuito)  = {a_circuito:.8f}")
    print(f"a (teorica)   = {a_teorica:.8f}")
    assert abs(a_circuito - a_teorica) < 1e-10, "El circuito no codifica sin^2(theta(f))."
    print("OK: el circuito implementa exactamente la codificacion sin^2.")

    # (b) post_processing(a) debe devolver (aprox.) la esperanza del coste.
    #     El residuo es el sesgo de la aproximacion del seno, de orden c^2.
    e_recuperada = float(F.post_processing(a_circuito))
    e_exacta = esperanza_ventana(problema, i, xi)
    rango = float(f_vals.max() - f_vals.min())
    print(f"esperanza recuperada (post-proceso) = {e_recuperada:.6f}")
    print(f"esperanza exacta (ventana)          = {e_exacta:.6f}")
    print(f"sesgo de codificacion = {abs(e_recuperada - e_exacta):.6f} "
          f"({100 * abs(e_recuperada - e_exacta) / rango:.3f}% del rango)")
    assert abs(e_recuperada - e_exacta) <= 0.02 * rango + 1e-9
    print("OK: el post-procesado recupera la esperanza (sesgo ~c^2 controlado).")


def paso_3_iae(problema, i, valores_xi):
    print("\n--- PASO 3: Amplitude Estimation completo vs analitico ---")
    for xi in valores_xi:
        t0 = time.perf_counter()
        e_ae = esperanza_ae(problema, i, xi)
        dt = time.perf_counter() - t0
        e_vent = esperanza_ventana(problema, i, xi)
        e_full = esperanza_analitica(problema, i, xi)
        print(f"zona {i}, xi={xi:2d}:  AE={e_ae:8.4f}  ventana={e_vent:8.4f}  "
              f"analitica(cuantil)={e_full:8.4f}  [{dt:.1f}s]")
        # Tolerancia: sesgo de codificacion (~2% rango) + epsilon del IAE.
        demandas, probs = problema.distribucion_ventana(i)
        f_vals = problema.coste(i, demandas, xi)
        rango = float(np.max(f_vals) - np.min(f_vals)) or 1.0
        assert abs(e_ae - e_vent) <= 0.03 * rango + 1e-6, \
            f"AE se desvia demasiado del analitico en xi={xi}."
    print("OK: AE coincide con el calculo analitico dentro de la tolerancia.")


def paso_4_resolucion(problema):
    print("\n--- PASO 4: resolucion greedy con AE vs analitico ---")
    x_ana, coste_ana = asignar_greedy(problema, esperanza_analitica)
    print(f"greedy + analitico : x = {x_ana}, coste = {coste_ana:.4f}")

    t0 = time.perf_counter()
    estimador_ae = crear_esperanza_ae(problema)
    x_ae, coste_ae = asignar_greedy(problema, estimador_ae)
    dt = time.perf_counter() - t0
    print(f"greedy + AE        : x = {x_ae}, coste = {coste_ae:.4f}  [{dt:.1f}s]")

    print("Asignaciones iguales:", bool(np.array_equal(x_ana, x_ae)))
    print("Factible:", problema.es_factible(x_ae),
          "| conductores:", x_ae.sum(), "/", problema.C)
    return x_ana, x_ae


def main():
    parametros_poisson = [
        {"mu": 5},
        {"mu": 8},
        {"mu": 3},
        {"mu": 6}
    ]

    problema_abs = ProblemaAbsoluto(
        C=20,
        tipo_demanda="poisson",
        parametros_demanda=parametros_poisson
    )

    problema_pen = ProblemaPenalizado(
        C=20,
        tipo_demanda="poisson",
        parametros_demanda=parametros_poisson,
        gamma=[10, 8, 12, 9],
        beta=[2, 3, 2, 4]
    )

    print("=" * 70)
    print("VALIDACION INCREMENTAL DEL MODULO DE AMPLITUDE ESTIMATION")
    print("=" * 70)

    # Validacion sobre el problema absoluto (zona 1, la de mayor demanda).
    paso_1_carga(problema_abs, i=1)
    paso_2_operador_A(problema_abs, i=1, xi=7)
    paso_3_iae(problema_abs, i=1, valores_xi=[0, 5, 7, 10])

    # El mismo modulo funciona sin cambios con el coste penalizado
    # (pendientes asimetricas gamma != beta).
    paso_2_operador_A(problema_pen, i=0, xi=5)
    paso_3_iae(problema_pen, i=0, valores_xi=[0, 5, 8])

    # Resolucion completa del problema con el estimador cuantico.
    paso_4_resolucion(problema_abs)


if __name__ == "__main__":
    main()
