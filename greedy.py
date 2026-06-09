import numpy as np

def execute(instance):
    n = instance["n"]
    p = instance.get("p", instance.get("m"))
    d = np.array(instance["d"])

    # Paso 0: Calcular g(i) = suma de distancias a todos los demás
    g = d.sum(axis=1)

    # Paso 1: Encontrar los p elementos con mayor g(i)
    selected = [int(x) for x in np.argsort(g)[-p:][::-1]]# índices de mayor a menor

    # Calcular la diversidad total del subconjunto
    total_diversity = 0
    for i in range(p-1):
        for j in range(i+1, p):
            total_diversity += d[selected[i], selected[j]]
    
    solution = [x + 1 for x in selected]

    return solution, total_diversity