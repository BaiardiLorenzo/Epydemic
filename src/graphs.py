import random

import networkx as nx

from src.config import init_infect
from src.infection import init_infected


def cycle_graph_test(nodes: int = 20) -> (nx.Graph, nx.Graph):
    """
    Test the cycle graph

    :param nodes: number of nodes
    :return: Physical and Virtual graph
    """
    PG = nx.cycle_graph(nodes)
    VG = nx.cycle_graph(nodes)
    init_infected(PG, init_infect)
    return PG, VG


def scale_free_graph_test(nodes: int = 20, mPG: int = 6, mVG: int = 6) -> (nx.Graph, nx.Graph):
    """
    Test the scale free graph

    :param nodes: number of nodes
    :param mPG: average degree of the physical graph
    :param mVG: average degree of the virtual graph
    :return: Physical and Virtual graph
    """
    PG = nx.barabasi_albert_graph(nodes, mPG)
    VG = nx.barabasi_albert_graph(nodes, mVG)
    init_infected(PG, init_infect)
    return PG, VG


def random_graph_test(nodes: int = 20, pPG: float = 0.5, pVG: float = 0.5) -> (nx.Graph, nx.Graph):
    """
    Test the random graph

    :param nodes: number of nodes
    :param pPG: probability of the connection in the physical graph
    :param pVG: probability of the connection in the virtual graph
    :return: Physical and Virtual graph
    """
    PG = nx.gnp_random_graph(nodes, pPG)
    VG = nx.gnp_random_graph(nodes, pVG)
    init_infected(PG, init_infect)
    return PG, VG


def get_information_graph(PG: nx.Graph, VG: nx.Graph, q: float = 0.5) -> nx.DiGraph:
    """
    Create the information graph from the two graphs Physical and Virtual graph

    :param PG: Physical graph
    :param VG: Virtual graph
    :param q: q value
    :return: return the information graph
    """
    IG = nx.DiGraph()

    for node in PG.nodes():
        IG.add_node(node)

        # Add outgoing links from the physical graph with probability 1 - q
        for neighbor in PG.neighbors(node):
            if random.random() > q:
                IG.add_edge(node, neighbor)

        # Add links from the virtual graph with probability q
        for neighbor in VG.neighbors(node):
            if random.random() <= q:
                IG.add_edge(node, neighbor)

    return IG
