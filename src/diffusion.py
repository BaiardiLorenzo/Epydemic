import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from src.infection import init_infected, update_risk, new_infected, change_state, get_infected
from src.plot import plot_critical_j, plot_update
from src.utils import healthy, t_test, j_test, critical_j_plot


def critical_j_test(G: nx.Graph,
                    ts: np.array,
                    js: np.array,
                    rec_prob: float = 0.2,
                    immunity: str = healthy,
                    plot: bool = False):
    """
    Get the critical J values test

    :param G: graph
    :param ts: values of tau
    :param js: values of risk perception J
    :param rec_prob: recovery probability
    :param immunity: if the nodes have immunity
    :param plot: if the graph should be displayed
    :return: return the critical J values
    """
    results = {t_test: [], j_test: []}
    for t in ts:
        for j in js:
            print(f"t: {t}, j: {j}")
            v = diffusion(G.copy(), j, t, rec_prob=rec_prob, immunity=immunity, plot=plot)
            if v <= 0:
                results[t_test].append(t)
                results[j_test].append(j)
                break
        print("--------------------------------------------------", end="\n\n")
    plot_critical_j(results, file=critical_j_plot, prediction=False)


def diffusion(G: nx.Graph,
              J: float,
              t: float,
              rec_prob: float,
              iteration: int = 50,
              infected_nodes: int = 1,
              immunity: str = healthy,
              plot: bool = False,
              ) -> int:
    """
    Propagate the disease in the graph G

    :param G: graph
    :param J: perception risk
    :param t: bare infection probability
    :param rec_prob: prob of recovering at each step
    :param iteration: number of iterations, default 10
    :param infected_nodes: number of infected nodes, default 2
    :param immunity: recovered if immunity is possible
    :param plot: if the graph should be plotted
    :return: return the number of infected nodes
    """
    init_infected(G, infected_nodes)

    pos = nx.spring_layout(G)
    plt.figure()

    for i in range(iteration):
        if plot:
            plot_update(G, pos)
        update_risk(G, J, t)
        G = new_infected(G)
        G = change_state(G, rec_prob, immunity)
        print(f"Step: {i + 1}, Infected: {get_infected(G)}")
        print(G.nodes.data())
        print("\n")

    return get_infected(G)
