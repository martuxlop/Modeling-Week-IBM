# problema.py

import numpy as np

 
class ProblemaAbsoluto:
    """
    Problema 1:

        min E[ sum_i |x_i - d_i| ]

        s.a.
            sum_i x_i <= C
            x_i entero no negativo
    """

    def __init__(self, C, tipo_demanda, parametros_demanda):

        self.C = int(C)
        self.tipo_demanda = tipo_demanda
        self.parametros_demanda = parametros_demanda
        self.n = len(parametros_demanda)

    def es_factible(self, x):
        x = np.array(x)

        return (
            len(x) == self.n
            and np.all(x >= 0)
            and np.all(x.astype(int) == x)
            and np.sum(x) <= self.C
        )

class ProblemaPenalizado:
    """
    Problema 2:

        min (1/n) E[ sum_i gamma_i max(d_i - x_i, 0)
                   + beta_i max(x_i - d_i, 0) ]

        s.a.
            sum_i x_i <= C
            x_i entero no negativo
    """

    def __init__(self, C, tipo_demanda, parametros_demanda, gamma, beta):

        self.C = int(C)

        self.tipo_demanda = tipo_demanda
        self.parametros_demanda = parametros_demanda

        self.gamma = np.array(gamma, dtype=float)
        self.beta = np.array(beta, dtype=float)

        self.n = len(parametros_demanda)

        if len(self.gamma) != self.n or len(self.beta) != self.n:
            raise ValueError("gamma y beta deben tener longitud n.")

    def es_factible(self, x):
        x = np.array(x)

        return (
            len(x) == self.n
            and np.all(x >= 0)
            and np.all(x.astype(int) == x)
            and np.sum(x) <= self.C
        )