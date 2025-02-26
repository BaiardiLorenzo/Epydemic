import random
import networkx as nx
import numpy as np

from src.config import max_j
from src.infection import infected_prob, get_infected_neighbors, risk_perception, \
    prob_being_infected, get_percentage_infected
from src.utils import *

# ______________________________________________________________________________________________________________________
# Simulating the J percolation


def simulated_j_percolation(G: nx.Graph, tau: float, J: float, T: int) -> float:
    """
    Simulated J percolation using the formula for the probability of being infected:
    {1 - (1 - u(s, k))^s} = p(s, k)


    :param G: graph just infected
    :param tau: bare infection probability
    :param J: risk perception
    :param T: number of iterations
    :return: the percentage of infected nodes
    """
    # Get the states of the nodes
    states = {node: G.nodes[node][state] for node in G.nodes}

    for _ in range(T):
        c_states = states
        for node in G.nodes:
            k = G.degree(node)
            # TODO: implementation of the annealed version
            s = get_infected_neighbors(G, node, c_states)  # quenched version
            r = random.random()
            psk = prob_being_infected(s, k, tau, J)
            states[node] = infected if r < psk else healthy
    return get_percentage_infected(G, states)


def simulated_approx_j_percolation(G: nx.Graph, tau: float, J: float, T: int) -> float:
    """
    Simulated J percolation with approximation of the formula for the probability of being infected:
    {t * exp(-J * s / k)} = u(s, k)

    :param G: graph just infected
    :param tau: bare infection probability
    :param J: risk perception
    :param T: number of iterations
    :return: return the percentage of infected nodes
    """
    for _ in range(T):
        prev_states = {node: G.nodes[node][state] for node in G.nodes}
        for node in G.nodes:
            k = G.degree(node)
            # TODO: implementation the annealed version
            s = get_infected_neighbors(G, node, prev_states)  # quenched version
            r = random.random()
            G.nodes[node][state] = infected if r < s * infected_prob(s, k, tau, J) else healthy
    return get_percentage_infected(G)


# ______________________________________________________________________________________________________________________
# Simple Percolation Tests (Direct Percolation)


def simulated_simple_percolation(G: nx.graph, T: int, tau: float) -> float:
    """
    Simulated simple percolation:
    TODO add formula for simple percolation
    xi(t+1) =

    :param G: graph
    :param T: number of iterations
    :param tau: infection probability
    :return: return the percentage of infected nodes
    """
    states = {node: G.nodes[node][state] for node in G.nodes}

    for _ in range(T):
        c_state = states.copy()
        for node in G.nodes:
            caution = any(tau > random.random() and c_state[n] == infected for n in G.neighbors(node))
            states[node] = infected if caution else healthy
    return get_percentage_infected(G, states)


def tau_simple_percolation(G: nx.Graph, iterations: int) -> float:
    """
    Update the values of tau parameters associated with the nodes in graph G based on simple percolation.

    taui[t+1] = min max(rij(t), taui(t)) calculated for each neighbor of node i

    This formula updates the value tau[i] as indicated in the simple percolation section.
    Tau is a vector such that the i-th component of tau corresponds to the tau value for node i.
    We start from the scenario where all nodes are infected, so tau is a vector of zeros.
    I copy the tau vector into a vector ctau to avoid overwriting the value of tau at each iteration of the min max.
    The nonzero function returns indices corresponding to the neighbors of node i.
    M is the vector of max values from which I extract the minimum.

    :param G: NetworkX graph
    :param iterations: Number of iterations
    :return: Updated tau values for each node
    """
    # Initialize tau values for each node
    tau = {node: 0.0 for node in G.nodes()}

    for _ in range(iterations):
        ct = tau.copy()
        # Iterate over each node in the graph
        for node in G.nodes():
            # Calculate max values for all neighbors
            # Update tau[node] using the minimum of max_values

            # NOTFIXME: Check if necessary to append 1 to max_values before finding the min
            tau[node] = np.min([1]+[max(random.random(), ct[j]) for j in G.neighbors(node)])
    return min(tau.values())

# ______________________________________________________________________________________________________________________
# Infection with risk percolation


# FIXME Check if the function is working properly
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
    # j_values = {node: float('inf') for node in G.nodes()}
    j_values = {node: max_j for node in G.nodes()}

    for _ in range(T):
        cj = j_values.copy()
        for node in G.nodes():
            k = G.degree(node)
            s = np.array([sum(cj[n] >= cj[j] for n in G.neighbors(node)) for j in G.neighbors(node)])
            jp = np.array([risk_perception(k, si, np.random.random(), tau) for si in s])
            cjs = np.array([cj[j] for j in G.neighbors(node)])
            j_values[node] = max(np.append(0, np.minimum(cjs, jp)))
    return max(j_values.values())


# ______________________________________________________________________________________________________________________
# The self-organized percolation method for multiplex networks

# FIXME Check if the function is working properly
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
    # j_values = {node: float('inf') for node in PG.nodes()}
    j_values = {node: max_j for node in PG.nodes()}

    for _ in range(T):
        cj = j_values.copy()
        for node in PG.nodes():
            k = PG.degree(node)
            s = np.array([sum(cj[n] >= cj[j] for n in IG.neighbors(node)) for j in PG.neighbors(node)])
            jp = np.array([risk_perception(k, si, np.random.random(), tau) for si in s])
            cjs = np.array([cj[j] for j in PG.neighbors(node)])
            j_values[node] = max(np.append(0, np.minimum(cjs, jp)))
    return max(j_values.values())
