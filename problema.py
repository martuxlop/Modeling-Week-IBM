import numpy as np


class ProblemaUber:
    """
    Problema:

        min (1/n) * sum_i E[
            gamma_i * max(d_i - x_i, 0)
            + beta_i * max(x_i - d_i, 0)
        ]

        s.a. sum_i x_i <= C
             x_i >= 0
             x_i entero
    """

    def __init__(self, C, gamma, beta, demanda_media):
        self.C = int(C)
        self.gamma = np.array(gamma, dtype=float)
        self.beta = np.array(beta, dtype=float)
        self.demanda_media = np.array(demanda_media, dtype=float)

        self.n = len(self.demanda_media)

        assert len(self.gamma) == self.n
        assert len(self.beta) == self.n

    def coste_escenario(self, x, d):
        """
        Calcula el coste para una asignación x y una demanda d.
        """
        x = np.array(x, dtype=float)
        d = np.array(d, dtype=float)

        falta_demanda = np.maximum(d - x, 0)
        exceso_conductores = np.maximum(x - d, 0)

        coste = self.gamma * falta_demanda + self.beta * exceso_conductores

        return np.mean(coste)

    def es_factible(self, x):
        """
        Comprueba si x cumple sum_i x_i <= C y x_i >= 0.
        """
        x = np.array(x)

        return (
            np.all(x >= 0)
            and np.all(x.astype(int) == x)
            and np.sum(x) <= self.C
        )