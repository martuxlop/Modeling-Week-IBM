import numpy as np
def asignar_greedy(problema, esperanza_zona):
    n, C = problema.n, problema.C
    x = np.zeros(n, dtype=int)

    coste_actual = np.array([esperanza_zona(problema, i, 0) for i in range(n)])
    coste_sig    = np.array([esperanza_zona(problema, i, 1) for i in range(n)])
    ganancia     = coste_actual - coste_sig

    for _ in range(C):
        i = int(np.argmax(ganancia))
        if ganancia[i] <= 0:            
            break
        x[i] += 1
        coste_actual[i] = coste_sig[i]
        coste_sig[i]    = esperanza_zona(problema, i, x[i] + 1)  # solo recalcula la zona tocada
        ganancia[i]     = coste_actual[i] - coste_sig[i]

    return x, float(np.mean(coste_actual))