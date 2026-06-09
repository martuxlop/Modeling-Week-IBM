# problema.py

import numpy as np


class Problema:

    def __init__(self, C, demanda_media):

        self.C = int(C)

        self.demanda_media = np.array(
            demanda_media,
            dtype=float
        )

        self.n = len(demanda_media)

    def es_factible(self, x):

        x = np.array(x)

        return (
            len(x) == self.n
            and np.all(x >= 0)
            and np.all(x.astype(int) == x)
            and np.sum(x) <= self.C
        )