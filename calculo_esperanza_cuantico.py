# calculo_esperanza_cuantico.py
"""
Cálculo de la esperanza del coste por zona mediante Amplitude Estimation (AE).

Idea general
------------
Para una zona i y una asignación xi fijas, queremos estimar

    E[f(D)] = sum_j p_j * f(d_j),

donde D es la demanda (discretizada en una ventana finita) y f es el coste de
la zona, que en este problema es EXACTAMENTE lineal a trozos:

    f(d) = gamma_i * max(d - xi, 0) + beta_i * max(xi - d, 0)

(el problema absoluto es el caso gamma_i = beta_i = 1).

El algoritmo cuántico convierte "calcular una esperanza" en "estimar la
probabilidad de medir |1> en un qubit". Se construye un operador A en dos
pasos (esquema de Woerner-Egger, "Quantum Risk Analysis"):

1.  Carga de distribución P sobre el registro índice (n qubits):

        P|0...0> = sum_j sqrt(p_j) |j>

    El estado base |j> representa la demanda d = d_lo + j (la ventana puede
    no empezar en 0; d_lo es el primer valor del soporte). La pmf vive en
    las amplitudes: medir el registro daría j con probabilidad p_j.

2.  Codificación del coste F sobre un qubit objetivo (ancilla):

        F|j>|0> = |j> ( sqrt(1 - f~(d_j)) |0> + sqrt(f~(d_j)) |1> )

    donde f~ es el coste normalizado a [0, 1]. En la práctica F no codifica
    f~ exactamente sino sin^2(pi/4 + (pi*c/2) * (f~ - 1/2)), casi lineal en
    f~ para c (rescaling_factor) pequeño; el `post_processing` del propio
    operador invierte esta transformación y la normalización.

Componiendo A = F · (P ⊗ I), la probabilidad de medir |1> en el objetivo es

    a = P(objetivo = 1) = sum_j p_j * sin^2(theta(d_j)) ≈ E[f~(D)] (tras c).

Amplitude Estimation estima esa amplitud a. Usamos la variante ITERATIVA
(IterativeAmplitudeEstimation, Grinko et al.), que no añade qubits de
evaluación (a diferencia del AE canónico con estimación de fase): controla
la precisión epsilon repitiendo potencias del operador de Grover
Q = A S0 A† Sψ, manteniendo el speedup O(1/epsilon) frente al O(1/epsilon^2)
de Monte Carlo.

Recuento de qubits del circuito:
    n          = ceil(log2(K))  qubits de índice (K = tamaño de la ventana)
    1          qubit objetivo
    #tramos-1  ancillas del comparador (1 si el breakpoint xi cae dentro
               del dominio, 0 si el coste es una sola recta)

Decisiones de diseño del módulo (acordadas para el proyecto):
    - Un AE POR DISTRITO (no conjunto): el greedy necesita esperanzas
      marginales por zona, la demanda es independiente entre zonas y el
      objetivo es separable; el AE conjunto necesitaría n_zonas x n qubits
      y una normalización global que degrada la precisión relativa.
    - c = 0.25 fijo por defecto (sesgo ~c^2 corregido en post-proceso).
    - La distribución llega YA DISCRETIZADA (ventana media ± k·sigma,
      `Problema.distribucion_ventana`), desacoplando discretización y AE.
    - Misma firma que el resto de métodos: esperanza_zona(problema, i, xi),
      vía la fábrica `crear_esperanza_ae` (que además memoiza por (i, xi),
      porque cada llamada AE construye y simula un circuito).
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearAmplitudeFunctionGate, StatePreparation
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation


# Hiperparámetros por defecto del módulo.
C_RESCALADO = 0.25     # rescaling_factor c: compromiso sesgo (~c^2) vs coste (~1/c)
EPSILON = 2e-3         # precisión objetivo sobre la amplitud a (1e-3 dispara el
                       # nº de potencias de Grover y el tiempo de simulación)
ALPHA = 0.05           # 1 - alpha = nivel de confianza del intervalo de IAE
SEED = 170             # semilla del sampler (reproducibilidad)
SHOTS = 2048           # shots por circuito en cada ronda de IAE


def _tramos_coste(problema, i, xi, d_lo, d_fin):
    """
    Describe el coste de la zona i como función lineal a trozos sobre el
    dominio [d_lo, d_fin], en la convención de LinearAmplitudeFunctionGate:
    en el tramo que empieza en el breakpoint b_k,

        f(d) = slope_k * (d - b_k) + offset_k,

    es decir, offset_k es el VALOR de f en el breakpoint (convención
    verificada empíricamente contra el statevector).

    El coste es f(d) = gamma*max(d-xi,0) + beta*max(xi-d,0), una "V" con
    vértice en xi. Según dónde caiga xi respecto a la ventana hay 3 casos:

    - xi <= d_lo : solo se ve la rama derecha (creciente). Una recta.
    - xi >= d_fin: solo se ve la rama izquierda (decreciente). Una recta.
    - en medio   : dos tramos con breakpoint en xi.

    Los dos primeros casos aparecen de forma natural durante el greedy
    (siempre evalúa xi = 0 y xi = 1, y puede superar la ventana por arriba).
    """
    gamma = float(problema.gamma[i])
    beta = float(problema.beta[i])

    if xi <= d_lo:
        # Rama derecha: f(d) = gamma * (d - xi), creciente en todo el dominio.
        return [d_lo], [gamma], [gamma * (d_lo - xi)]

    if xi >= d_fin:
        # Rama izquierda: f(d) = beta * (xi - d), decreciente en todo el dominio.
        return [d_lo], [-beta], [beta * (xi - d_lo)]

    # Caso general: vértice dentro del dominio, f(xi) = 0.
    return [d_lo, xi], [-beta, gamma], [beta * (xi - d_lo), 0.0]


def construir_operador_A(problema, i, xi, demandas, probs, c=C_RESCALADO):
    """
    Construye el operador A = F · (P ⊗ I) para la zona i y asignación xi.

    Parámetros
    ----------
    demandas, probs : soporte contiguo (paso 1) y pmf renormalizada de la
        demanda, p. ej. de `problema.distribucion_ventana(i)`.
    c : rescaling_factor de la codificación del coste.

    Devuelve
    --------
    (A, F, n, exacto):
        A      : QuantumCircuit del operador completo (None si trivial).
        F      : LinearAmplitudeFunctionGate, cuya `post_processing` invierte
                 la codificación (None si trivial).
        n      : nº de qubits del registro índice.
        exacto : si el coste es constante sobre el dominio no hace falta
                 circuito; se devuelve aquí el valor exacto (None si no).
    """
    d_lo = int(demandas[0])
    K = len(demandas)

    # ---- Registro índice: n qubits, dominio con padding hasta 2^n ----------
    # K valores de demanda requieren n = ceil(log2(K)) qubits. Los estados
    # j >= K se rellenan con probabilidad 0 (no contribuyen a la esperanza),
    # pero SÍ forman parte del dominio sobre el que se define F, por lo que
    # la normalización del coste debe calcularse sobre todo [d_lo, d_fin].
    n = max(1, int(np.ceil(np.log2(K))))
    N = 2 ** n
    d_fin = d_lo + N - 1

    p = np.zeros(N)
    p[:K] = probs
    p = p / p.sum()

    # ---- Normalización del coste a [0, 1] ----------------------------------
    # Imagen (f_min, f_max) calculada numéricamente con problema.coste sobre
    # todo el dominio (incluido el padding): robusto y coherente por
    # construcción con los tramos analíticos de _tramos_coste.
    grid = np.arange(d_lo, d_fin + 1)
    f_vals = np.asarray(problema.coste(i, grid, int(xi)), dtype=float)
    f_min, f_max = float(f_vals.min()), float(f_vals.max())

    if np.isclose(f_min, f_max):
        # Coste constante: la esperanza es exacta, no hay nada que estimar.
        return None, None, n, f_min

    # ---- Carga de distribución P: amplitudes sqrt(p_j) ---------------------
    amplitudes = np.sqrt(p)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    carga = StatePreparation(amplitudes)

    # ---- Codificación del coste F ------------------------------------------
    breakpoints, slopes, offsets = _tramos_coste(problema, i, int(xi), d_lo, d_fin)

    F = LinearAmplitudeFunctionGate(
        n,
        slope=slopes,
        offset=offsets,
        domain=(d_lo, d_fin),     # mapeo afín |j> -> d = d_lo + j (paso 1)
        image=(f_min, f_max),
        breakpoints=breakpoints,
        rescaling_factor=c,
    )

    # ---- Composición A = F · (P ⊗ I) ---------------------------------------
    # Orden de qubits de F: [0..n-1] índice, [n] objetivo, [n+1..] ancillas
    # del comparador del breakpoint.
    A = QuantumCircuit(F.num_qubits)
    A.append(carga, range(n))
    A.append(F, range(F.num_qubits))

    return A, F, n, None


def esperanza_ae(problema, i, xi, demandas=None, probs=None,
                 c=C_RESCALADO, epsilon=EPSILON, alpha=ALPHA,
                 seed=SEED, shots=SHOTS):
    """
    Esperanza del coste de la zona i para la asignación xi, estimada con
    Iterative Amplitude Estimation.

    Si no se pasan (demandas, probs), usa la discretización por ventana del
    propio problema. `epsilon` y `alpha` controlan el estimador: error <=
    epsilon sobre la amplitud con confianza 1 - alpha. OJO: el error sobre
    la ESPERANZA escala como epsilon * (f_max - f_min) * 2/(pi*c) tras el
    post-procesado; con costes grandes conviene epsilon pequeño.

    Devuelve un float (la esperanza estimada, ya des-normalizada).
    """
    if demandas is None or probs is None:
        demandas, probs = problema.distribucion_ventana(i)

    A, F, n, exacto = construir_operador_A(problema, i, xi, demandas, probs, c)

    if exacto is not None:
        return float(exacto)

    # EstimationProblem empaqueta: el circuito A, qué qubit es el objetivo
    # (el n, justo tras el registro índice) y cómo deshacer la codificación.
    problem = EstimationProblem(
        state_preparation=A,
        objective_qubits=[n],
        post_processing=F.post_processing,
    )

    # IAE: refina iterativamente un intervalo de confianza para el ángulo
    # theta (a = sin^2(theta)) aplicando potencias crecientes del operador
    # de Grover. Sin qubits de evaluación; precisión vía repeticiones.
    sampler = StatevectorSampler(seed=seed, default_shots=shots)
    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon,
        alpha=alpha,
        sampler=sampler,
    )

    resultado = iae.estimate(problem)

    # estimation_processed = post_processing(a_estimada): invierte el sin^2,
    # el factor c y la normalización [f_min, f_max] -> esperanza del coste.
    return float(resultado.estimation_processed)


def crear_esperanza_ae(problema, c=C_RESCALADO, epsilon=EPSILON, alpha=ALPHA,
                       k_izq=3.0, k_der=4.0, seed=SEED, shots=SHOTS):
    """
    Fábrica del estimador cuántico con la firma estándar del proyecto:

        esperanza_zona(problema, i, xi) -> float

    de modo que es intercambiable con `esperanza_analitica` y con el
    estimador Monte Carlo en `asignar_greedy`, `coste_total`, etc.

    Además memoiza los resultados por (i, xi): cada evaluación AE construye
    y simula un circuito (cara), y tanto el greedy como `coste_total`
    reevalúan pares (i, xi) ya visitados.
    """
    cache = {}

    def esperanza(problema_, i, xi):
        xi = int(xi)
        if (i, xi) not in cache:
            demandas, probs = problema.distribucion_ventana(i, k_izq, k_der)
            cache[(i, xi)] = esperanza_ae(
                problema, i, xi, demandas, probs,
                c=c, epsilon=epsilon, alpha=alpha, seed=seed, shots=shots,
            )
        return cache[(i, xi)]

    return esperanza
