import networkx as nx
import numpy as np

from src.config import zero_threshold
from src.infection import get_average_graph_degree, simulated_j_percolation, simulated_approx_j_percolation
from src.plot import plot_critical_j
from src.utils import infected, state, healthy, t_test, j_test, j_pred, not_approx_jc_plot, approx_plot_jc_plot


def not_approx_critical_j_test(G: nx.Graph, T: int, ts: np.array, js: np.array):
    """
    Get the critical J values test

    :param G: graph
    :param T: iteration
    :param ts: values of tau
    :param js: values of risk perception J
    :return: return the critical J values
    """

    # Calculate the average degree of the graph
    k = get_average_graph_degree(G)

    results = {t_test: [], j_test: [], j_pred: []}
    for t in reversed(ts):
        # Calculate the critical J prediction from the mean field approximation
        jc_pred = k * np.log(k * t)
        print(f"Critical J prediction: {jc_pred}")
        for j in js:
            print(f"t: {round(t,2)}, j: {round(j,2)}")
            v = simulated_j_percolation(G.copy(), t, j, T)
            if v <= zero_threshold:
                results[t_test].append(t)
                results[j_test].append(j)
                results[j_pred].append(jc_pred if jc_pred > 0 else 0)
                break
        print("--------------------------------------------------", end="\n\n")
    plot_critical_j(results, file=not_approx_jc_plot)


def approx_critical_j_test(G: nx.Graph, T: int, ts: np.array, js: np.array):
    """
    Get the critical J values test

    :param G: graph
    :param T: iteration
    :param ts: values of tau
    :param js: values of risk perception J
    :return: return the critical J values
    """

    # Calculate the average degree of the graph
    k = get_average_graph_degree(G)

    results = {t_test: [], j_test: [], j_pred: []}
    for t in reversed(ts):
        jc_pred = k * np.log(k * t)
        print(f"Critical J prediction: {jc_pred}")
        for j in js:
            print(f"t: {round(t,2)}, j: {round(j,2)}")
            v = simulated_approx_j_percolation(G.copy(), t, j, T)
            if v <= zero_threshold:
                results[t_test].append(t)
                results[j_test].append(j)
                results[j_pred].append(jc_pred if jc_pred > 0 else 0)
                break
        print("--------------------------------------------------", end="\n\n")
    plot_critical_j(results, file=approx_plot_jc_plot)


def multiplex_percolation_critical_j_test(IG: nx.DiGraph, PG: nx.Graph, T: int, ts: np.array, js: np.array, qs: np.array):
    """
    Get the critical J values test

    :param IG: Information graph
    :param PG: Physical graph
    :param T: iteration
    :param ts: values of tau
    :param js: values of risk perception J
    :param qs: values of q
    :return: return the critical J values
    """
    results = {q_test: [], t_test: [], j_test: [], j_pred: []}
    for q in qs:
        for t in reversed(ts):
            # Calculate the jc prediction about percolation
            jc_pred = multiplex_percolation(IG, PG, t, T)
            print(f"Critical J prediction: {jc_pred}")
            for j in js:
                print(f"t: {round(t,2)}, j: {round(j,2)}")
                # Simulate the infection with the percolation scenario
                v = simulated_j_percolation(PG.copy(), t, j, T)
                if v <= zero_threshold:
                    results[q_test].append(q)
                    results[t_test].append(t)
                    results[j_test].append(j)
                    results[j_pred].append(jc_pred if jc_pred > 0 else 0)
                    break
            print("--------------------------------------------------", end="\n\n")


def simulated_multiplex(PG: nx.Graph,
                        VG: nx.Graph,
                        q: float,
                        tau: float,
                        J: float,
                        T: int) -> float:
    """
    Simulated J percolation

    :param PG: graph just infected
    :param VG: virtual graph
    :param q: connection probability for the information graph
    :param tau: bare infection probability
    :param J: risk perception
    :param T: number of iterations
    :return: return the percentage of infected nodes
    """
    IG = get_information_graph(PG, VG, q)
    for _ in range(T):
        cIG = IG.copy
        for node in PG.nodes:
            k = PG.degree(node)
            s = get_infected_neighbors(cIG, node)  # quenched version @TODO: implementation of the annealed version
            r = np.random.uniform(0, 1)
            psk = 1 - pow(1-infected_prob(s, k, tau, J), s)
            if r < psk:
                PG.nodes[node][state] = infected
                IG.nodes[node][state] = infected
            else:
                PG.nodes[node][state], IG.nodes[node][state] = healthy
    return get_infected(PG) / PG.number_of_nodes()


def critical_j_test():
    PG, VG = random_graph_test(n_nodes, pPG=prob_k, pVG=prob_k)
    critical_j_test(PG, ts, js, rec_prob=rec_prob, immunity=immunity, plot=False)


def critic_j_percolation(G: nx.Graph, tau: float, T: int) -> float:
    """
    Calculate the critical J values for percolation.

    I aim to determine the parameter J for which there is no long-term infection spread.
    [J < -(k/s) ln(r/τ)] is the condition for a node to become infected.
    In the function, I extract r and check if it equals 1.
    J[i] is the minimum value of J that causes node i to become infected.
    Skipping the calculations, I directly move to formula (15), which is the one I want to implement.
    Ji (t + 1) = max min (Jj(t), (-ki/si[Jj(t)]) * ln(rij(t)/τ)).
    si[J] is a function that sums [Jj(t) >= J] for all neighbors.

    :param G: NetworkX graph
    :param tau: Infection probability
    :param T: Number of iterations
    :return: Updated critical J values
    """

    # Initialize J values for each node
    # TODO evaluation the initialization of J values
    j_values = {node: float('inf') for node in G.nodes()}

    for _ in range(T):
        cj = j_values.copy()
        for node in G.nodes():
            m = []
            k = G.degree(node)
            for j in G.neighbors(node):
                # Calculate sum term
                s = sum(cj[n] >= cj[j] for n in G.neighbors(node))
                r = random.random()
                # Calculate min max term
                jp = risk_perception(k, s, r, tau)
                m.append(min(cj[j], jp))
            # Update J with the maximum value calculated
            j_values[node] = max(m+[0])
    return max(j_values.values())


def multiplex_percolation(IG: nx.DiGraph, PG: nx.graph, tau: float, T: int) -> float:
    """
    Multiplex percolation model

    :param IG: Information graph
    :param PG: Physical graph
    :param tau: Infection probability
    :param T: Number of iterations
    :return: Updated critical J values
    """

    # Initialize J values for each node
    j_values = {node: float('inf') for node in PG.nodes()}

    for _ in range(T):
        cj = j_values.copy()
        for node in PG.nodes():
            m = [0]
            k = PG.degree(node)  # Calculate the degree of node i
            for j in PG.neighbors(node):  # Iterate over neighbors of node i in PG
                # Count neighbors with J greater than J[j]
                si = sum(cj[z] >= cj[j] for z in IG.neighbors(node))
                # Calculate min max term
                r = random.random()
                jp = risk_perception(k, si, r, tau)
                m.append(min(cj[j], jp))
            j_values[node] = max(m)  # Update J[i] with the maximum value calculated
    return max(j_values.values())


def percolation_jc_test(G: nx.Graph,
                        T: int,
                        ts: np.array,
                        js: np.array) -> None:
    """
    Get the critical J values test

    :param G: graph
    :param T: iteration
    :param ts: values of tau
    :param js: values of risk perception J
    :return: return the critical J values
    """

    results = {t_test: [], j_test: [], j_pred: []}
    for t in reversed(ts):
        # Calculate the jc prediction about percolation
        jc_pred = critic_j_percolation(G, t, T)
        print(f"Percolation-Critical J prediction: {jc_pred}")
        for j in js:
            print(f"t: {round(t, 2)}, j: {round(j, 2)}")
            # Simulate the infection with the percolation scenario
            v = simulated_j_percolation(G, t, j, T)
            if v <= zero_threshold:
                results[t_test].append(t)
                results[j_test].append(j)
                results[j_pred].append(jc_pred)
                break
        print("--------------------------------------------------", end="\n\n")
    plot_critical_j2(results, file=approx_plot_jc_plot)


def percolation_jc_test_2(G: nx.Graph,
                          T: int,
                          ts: np.array) -> dict:
    """
    Get the critical J values test

    :param G: graph
    :param T: iteration
    :param ts: values of tau
    :return: return results of the test
    """

    results = {t_test: [], j_test: [], j_pred: []}
    for t in reversed(ts):
        # Calculate the jc prediction about percolation
        jc_pred = critic_j_percolation(G, t, T)
        print(f"Percolation-Critical J prediction: {round(jc_pred, 2)}")
        # print(f"t: {round(t,2)}, j: {round(jc_pred,2)}")
        js = np.arange(0, 200, 0.1)
        for j in js:
            print(f"t: {round(t, 2)}, j: {round(j, 2)}")
            # Simulate the infection with the percolation scenario
            v = simulated_j_percolation(G, t, j, T)
            if v <= zero_threshold:
                results[t_test].append(t)
                results[j_test].append(j)
                results[j_pred].append(jc_pred)
                break
    print("--------------------------------------------------", end="\n\n")
    return results


def percolation_jc_test_3(G: nx.Graph,
                          T: int,
                          ts: np.array) -> dict:
    """
    Get the critical J values test

    :param G: graph
    :param T: iteration
    :param ts: values of tau
    :return: return results of the test
    """

    results = {t_test: [], j_test: [], j_pred: []}
    print(f"Percolation-Critical J prediction: ", end="")
    for t in reversed(ts):
        # Calculate the jc prediction about percolation
        jc_pred = critic_j_percolation(G, t, T)
        print(f"t: {round(t, 2)}, j: {round(jc_pred, 2)}")
        results[t_test].append(t)
        results[j_test].append(jc_pred)
        results[j_pred].append(jc_pred)
    print("--------------------------------------------------", end="\n\n")
    return results


def percolation_test():
    results = {}

    PG, VG = random_graph_test(n_nodes, pPG=prob_k, pVG=prob_k)
    # TODO The percolation jc test and multiplex percolation is not working
    #graph_type = "Poisson <k>=" + str(k)
    #print(f"Graph type: {graph_type}")
    percolation_jc_test(PG, iterations, ts, js)
    #result = percolation_jc_test_3(PG, iterations, ts)
    #results[graph_type] = result
#
    #plot_critical_j(results, file=approx_plot_jc_plot, prediction=False)