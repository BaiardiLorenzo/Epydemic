import random

import networkx as nx
import numpy as np
import random as rd

from src.config import eps
from src.utils import *


def get_critical_j(k: int, t: float) -> float:
    """
    Get the theoretical critical J value
    To compute the critical J value, we use the following formula:
    {k * log(k * t)} = Jc

    :param k: degree of the graph
    :param t: bare infection probability
    :return: return the critical J value
    """
    t = max(t, eps)
    k = max(k, eps)
    jc = k * np.log(k * t)
    return max(jc, 0)


def risk_perception(k: int, si: int, r: float, tau: float) -> float:
    """
    Compute the risk perception
    To compute the risk perception, we use the following formula:
    {-k/si * log(r/tau)} = Jc

    :param k: average degree
    :param si: number of infected neighbors
    :param r: random number
    :param tau: infection probability
    :return: risk perception
    """
    tau = max(tau, eps)
    si = max(si, eps)
    jc = -(k / si) * np.log(r / tau)
    return max(jc, 0)


def infected_prob(s: int, k: int, t: float, J: float) -> float:
    """
    Compute the probability of being infected
    To calculate the probability of being infected, we use the following formula:
    {t * exp(-J * s / k)} = u(s, k)

    :param s: number of infected neighbors
    :param k: degree of the node
    :param t: bare infection probability
    :param J: perception risk
    :return: return the probability of being infected
    """
    t = max(t, eps)
    J = max(J, eps)
    k = max(k, eps)
    u = t * np.exp(-J * s / k)
    return max(u, 0)


def prob_being_infected(s: int, k: int, tau: float, J: float) -> float:
    """
    Compute the probability of being infected
    To compute the probability of being infected, we use the following formula:
    {1 - (1 - u(s, k))^s} = p(s, k)

    :param s: number of infected neighbors
    :param k: degree of the node
    :param tau: bare infection probability
    :param J: perception risk
    :return: return the probability of being infected
    """
    return 1 - pow(1-infected_prob(s, k, tau, J), s)


def init_infected(G: nx.Graph, n: int = 1) -> None:
    """
    Initialize the graph G with n infected nodes

    :param G: graph
    :param n: number of infected nodes, default 1
    """
    infected_nodes = set(rd.sample(range(G.number_of_nodes()), n))
    for node in G.nodes:
        G.nodes[node][state] = infected if node in infected_nodes else healthy


def get_infected(G: nx.Graph, states: dict = None) -> int:
    """
    Count the number of infected nodes in a graph G

    :param G: graph
    :param states: states of the nodes
    :return: return the number of infected nodes
    """
    if states:
        return sum(states[node] == infected for node in G.nodes)
    else:
        return sum(G.nodes[node][state] == infected for node in G.nodes)


def get_percentage_infected(G: nx.Graph, states: dict = None) -> float:
    """
    Compute the percentage of infected nodes in the graph G

    :param G: graph
    :param states: states of the nodes
    :return: return the percentage of infected nodes
    """
    if states:
        return get_infected(G, states) / G.number_of_nodes()
    else:
        return get_infected(G) / G.number_of_nodes()


def get_infected_neighbors(G: nx.Graph, node: int, states: dict = None) -> int:
    """
    Count the number of infected neighbors of a node in the graph G

    :param G: graph
    :param node: list of neighbors
    :param states: states of the nodes
    :return: return the number of infected neighbors
    """
    if states:
        return sum(states[n] == infected for n in G.neighbors(node))
    else:
        return sum(G.nodes[n][state] == infected for n in G.neighbors(node))


def get_average_graph_degree(G: nx.Graph) -> int:
    """
    Compute the average degree of the graph G

    :param G: graph
    :return: value of the average degree of the graph
    """
    return int(sum(d for n, d in G.degree()) / G.number_of_nodes())


def update_risk(G: nx.Graph, J: float, t: float) -> None:
    """
    Function to evaluate the risk perception of each node in the graph G

    :param G: graph
    :param J: perception risk
    :param t: bare infection probability
    """
    for node in G.nodes:
        k = G.degree(node)
        s = get_infected_neighbors(G, node)
        G.nodes[node][risk] = prob_being_infected(s,k,t,J)
        #G.nodes[node][risk] = infected_prob(s, k, t, J)


def new_infected(G: nx.Graph) -> None:
    """
    Propagate the disease in the graph G with infected probability

    :param G: graph
    """
    for node in G.nodes:
        u = G.nodes[node][state]
        r = random.random()
        if r < u and G.nodes[node][state] == healthy:
            G.nodes[node][state] = infected


def change_state(G: nx.Graph, rec_prob: float, new_state: str) -> None:
    """
    Change the state of all nodes in the graph G

    :param G: graph
    :param rec_prob: recovery probability
    :param new_state: new state
    """
    for node in G.nodes:
        r = random.random()
        if G.nodes[node][state] == infected and r < rec_prob:
            G.nodes[node][state] = new_state
