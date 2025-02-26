import numpy as np
import networkx as nx

from src.config import zero_threshold
from src.graphs import get_information_graph
from src.infection import get_critical_j, get_average_graph_degree
from src.mean_field import simulated_mean_field_infection
from src.percolation import (tau_simple_percolation, simulated_simple_percolation, critic_j_percolation,
                             multiplex_percolation)
from src.utils import *


# ______________________________________________________________________________________________________________________
# Mean Field Tests


# TODO - Add the number of the infected nodes in the results
def mean_field_jc_test(k: int,
                       c: float,
                       T: int,
                       ts: np.array,
                       js: np.array) -> dict:
    """
    Get the critical J values test

    :param k: average degree of the graph
    :param c: initial percentage of infected nodes
    :param T: iteration
    :param ts: values of tau
    :param js: values of risk perception J
    :return: return the critical J values
    """

    results = {t_test: [], j_test: [], j_pred: []}
    for t in reversed(ts):
        jc_pred = get_critical_j(k, t)
        print(f"MF-Critical J prediction: {jc_pred}")
        for j in js:
            print(f"t: {round(t, 2)}, j: {round(j, 2)}, ", end="")
            v = simulated_mean_field_infection(k, t, c, T, j)
            print(f"c: {round(v, 2)}")
            if v <= zero_threshold:
                results[t_test].append(t)
                results[j_test].append(j)
                results[j_pred].append(jc_pred)
                break
        print("--------------------------------------------------", end="\n\n")
    return results


# ______________________________________________________________________________________________________________________
# Simple Percolation Tests (Direct Percolation)


def simple_tau_percolation_test(G: nx.Graph,
                                T: int,
                                ts: np.array) -> dict:
    """
    Get the value of tau_critical
    :param G: graph
    :param T: iteration
    :param ts: values of tau
    :return: return the critical tau values
    """
    tc_pred = tau_simple_percolation(G, iterations=T)

    results = {t_test: [], v_pred: [], t_pred: []}

    print(f"Percolation-Critical tau prediction: {tc_pred}")
    for t in reversed(ts):
        v = simulated_simple_percolation(G, T, tau=t)
        print(f"t: {round(t, 2)} Value of c: {v}")
        results[t_test].append(t)
        results[v_pred].append(v)
        results[t_pred].append(tc_pred)
        if v <= zero_threshold:
            break
    print("--------------------------------------------------", end="\n\n")
    return results


# ______________________________________________________________________________________________________________________
# Infection with risk percolation


def risk_percolation_j_test(G: nx.graph,
                            T: int,
                            ts: np.array) -> dict:
    """
    Get the critical J values test

    :param G:
    :param T:
    :param ts:
    :return:
    """
    results = {t_test: [], j_test: [], j_pred: []}
    k = get_average_graph_degree(G)
    for t in reversed(ts):
        jc_pred = critic_j_percolation(G, t, T)
        j = get_critical_j(k, t)
        results[t_test].append(t)
        results[j_test].append(j)
        results[j_pred].append(jc_pred)
        print(f"t: {round(t, 2)}, j: {round(j, 2)}, jc: {round(jc_pred, 2)}")
        print("--------------------------------------------------", end="\n\n")
    return results

# ______________________________________________________________________________________________________________________
# The self-organized percolation method for multiplex networks


def multiplex_percolation_jc_test(PG: nx.Graph,
                                  VG: nx.Graph,
                                  T: int,
                                  ts: np.array,
                                  qs: np.array) -> dict:
    """
    Get the critical J values test

    :param PG: Physical graph
    :param VG: Virtual graph
    :param T: iteration
    :param ts: values of tau
    :param qs: values of q
    :return: return the results of the test
    """

    results = {q_test: [], t_test: [], j_pred: []}
    for q in qs:
        IG = get_information_graph(PG, VG, q)
        for t in reversed(ts):
            print(f"q: {round(q, 2)}, t: {round(t, 2)}", end="")
            # Calculate the jc prediction about percolation
            jc_pred = multiplex_percolation(IG, PG, t, T)
            print(f" jc: {round(jc_pred, 2)}")
            results[q_test].append(q)
            results[t_test].append(t)
            results[j_pred].append(jc_pred)
        print("--------------------------------------------------", end="\n\n")
    return results
