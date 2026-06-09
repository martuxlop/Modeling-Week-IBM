from problema import ProblemaAbsoluto, ProblemaPenalizado
from greedy import asignar_greedy

from calculo_esperanza import (
    esperanza_analitica_1,
    esperanza_montecarlo_1,
    esperanza_analitica_2,
    esperanza_montecarlo_2
)


def mostrar_resultados(nombre, problema, esperanza):
    x, coste = asignar_greedy(problema, esperanza)

    print("\n" + "="*60)
    print(nombre)
    print("="*60)
    print("Asignación:", x)
    print("Conductores usados:", x.sum(), "/", problema.C)
    print("Coste esperado:", round(coste, 4))
    print("Factible:", problema.es_factible(x))


def main():

    parametros_poisson = [
        {"mu": 5},
        {"mu": 8},
        {"mu": 3},
        {"mu": 6}
    ]

    problema1 = ProblemaAbsoluto(
        C=20,
        tipo_demanda="poisson",
        parametros_demanda=parametros_poisson
    )

    problema2 = ProblemaPenalizado(
        C=20,
        tipo_demanda="poisson",
        parametros_demanda=parametros_poisson,
        gamma=[10, 8, 12, 9],
        beta=[2, 3, 2, 4]
    )

    mostrar_resultados(
        "PROBLEMA 1 - ANALÍTICO",
        problema1,
        esperanza_analitica_1
    )

    mostrar_resultados(
        "PROBLEMA 1 - MONTE CARLO",
        problema1,
        esperanza_montecarlo_1
    )

    mostrar_resultados(
        "PROBLEMA 2 - ANALÍTICO",
        problema2,
        esperanza_analitica_2
    )

    mostrar_resultados(
        "PROBLEMA 2 - MONTE CARLO",
        problema2,
        esperanza_montecarlo_2
    )


if __name__ == "__main__":
    main()