import networkx as nx
import scipy
from scipy import interpolate
from matplotlib import pyplot as plt
from src.config import *
from src.utils import *


# ______________________________________________________________________________________________________________________
# Plotting for Graphs


def plot_graph(G: nx.Graph, pos: dict, ax, oriented: bool = False, show: bool = False) -> None:
    """
    Plot the graph G

    :param G: graph
    :param ax: axis
    :param pos: position of the nodes
    :param oriented: if the graph is oriented
    :param show: if the graph should be displayed
    """
    if oriented:
        nx.draw(G, pos, ax=ax, node_color=blue, with_labels=True,
                arrows=True, arrowstyle='->', arrowsize=10)
    else:
        nx.draw(G, pos, ax=ax, node_color=blue, with_labels=True, )
    if show:
        plt.show()


def plot_all_graphs(PG: nx.Graph, VG: nx.Graph, IG: nx.Graph) -> None:
    """
    Plot the graphs PG, VG and IG

    :param PG: physical graph
    :param VG: virtual graph
    :param IG: information graph
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].set_title("Physical Graph")
    axs[1].set_title("Virtual Graph")
    axs[2].set_title("Information Graph")

    pos = nx.spring_layout(PG)

    plot_graph(PG, pos, ax=axs[0])
    plot_graph(VG, pos, ax=axs[1])
    plot_graph(IG, pos, ax=axs[2], oriented=True)

    plt.show()


def plot_update(G: nx.Graph, pos: dict) -> None:
    """
    Plot the updated graph G

    :param G: graph
    :param pos: position of the nodes
    """
    colors = [blue if G.nodes[node][state] == healthy else green if G.nodes[node][state] == recovered else red for node
              in G]
    nx.draw(G, node_color=colors, with_labels=True, pos=pos)
    plt.pause(1)


# ______________________________________________________________________________________________________________________
# Plotting for Critical J and Tau For mean field and percolation


def plot_critical_j(results: dict, file: str, prediction: bool = True) -> None:
    """
    Plot the critical J values

    :param results: dict of critical values
    :param file: file name
    :param prediction: if the prediction should be plotted
    """
    colors = [blue, red, green, black]  # Add more colors if needed

    plt.title("Infection model and mean-field approximation: N = " + str(n_nodes))
    for graph_type in results:
        color = colors.pop(0)
        plt.plot(results[graph_type][t_test], results[graph_type][j_test],
                 label=graph_type, color=color, marker="", linestyle="--", linewidth=3)
        if prediction:
            plt.plot(results[graph_type][t_test], results[graph_type][j_pred],
                     label="Theory: " + graph_type, color=black, marker="", linestyle="--", linewidth=1)
    plt.xlabel("τ")
    plt.ylabel("Jc")
    plt.legend()
    plt.savefig(path_plots + file)
    plt.show()


# ______________________________________________________________________________________________________________________
# Plotting Percolation for j values


def plot_percolation_critical_j(results: dict, file: str, prediction: bool = True) -> None:
    """
    Plot the critical J values

    :param results: dict of critical values
    :param file: file name
    :param prediction: if the prediction should be plotted
    """
    colors = [blue, red, green, black]  # Add more colors if needed

    plt.title("Percolation Jc and simulated Jc" + str(n_nodes))

    for graph_type in results:
        color = colors.pop(0)
        plt.plot(results[graph_type][t_test], results[graph_type][j_test],
                 label=graph_type, color=color, marker=".", linestyle="--")
        if prediction:
            plt.plot(results[graph_type][t_test], results[graph_type][j_pred],
                     label="Theory: " + graph_type, color=black, marker="", linestyle="--")
    plt.xlabel("τ")
    plt.ylabel("Jc")
    plt.legend()
    plt.savefig(path_plots + file)
    plt.show()


# ______________________________________________________________________________________________________________________
# Plotting Percolation for c and tau values


def plot_critical_t(results: dict, file: str, prediction: bool = True) -> None:
    """
    Plot the critical tau values

    :param results: dict of results
    :param file: file name
    :param prediction: if the prediction should be plotted
    """

    colors = [blue, red, green, black]  # Add more colors if needed
    plt.title("Percolation in single-layered networks: N = " + str(n_nodes))
    for graph_type in results:
        color = colors.pop(0)
        plt.plot(results[graph_type][t_test], results[graph_type][v_pred], label=graph_type, marker="", linestyle="-",
                 color=color)
        if prediction:
            plt.plot(results[graph_type][t_pred][0], 0, label="Theory τc", marker="s", linestyle=" ", color=color)
    plt.xlabel("τ")
    plt.ylabel("c")
    plt.legend()
    plt.savefig(path_plots + file)
    plt.show()


def plot_q_value(results: dict, file: str) -> None:
    """
    Plot the value of the Information Graph
    :param results:
    :param file:
    :return:
    """
    fig, ax = plt.subplots()
    ax.set(xlabel='τ', ylabel='q', title='Phase Diagram')

    x = np.array(results[t_test])
    y = np.array(results[q_test])
    z = np.array(results[j_pred])

    xi, yi = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]

    rbf = scipy.interpolate.Rbf(x, y, z)
    ai = rbf(xi, yi)

    img = ax.imshow(ai.T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])

    # Add points values
    # ax.scatter(x, y, c=z)

    bar = fig.colorbar(img)
    bar.set_label("Jc")

    plt.savefig(path_plots + file)
    plt.show()


def plot_t_value_percolation(critics: dict, file: str) -> None:
    """
    Plot the value of the Information Graph
    :param critics:
    :param file:
    :return:
    """
    plt.title("Tau values")
    keys = list(critics.keys())
    for i in keys:
        plt.plot(critics[i][q_test], critics[i][j_pred], label=f"τ: {i}", marker="o", linestyle="-")
    plt.xlabel("Q")
    plt.ylabel("Jc")
    plt.legend()
    plt.savefig(path_plots + file)
    plt.show()


# ______________________________________________________________________________________________________________________


def plot_comparison_j_q(results: dict, file: str) -> None:
    """
    Plot the comparison between J and Q values
    :param results:
    :param file:
    :return:
    """

    new_results = {t: {q_test: [], j_pred: []} for t in results[t_test]}
    for i in range(len(results[t_test])):
        new_results[results[t_test][i]][q_test].append(results[q_test][i])
        new_results[results[t_test][i]][j_pred].append(results[j_pred][i])

    plt.title("Multiplex Percolation - valuation with J and Q")
    # for t in new_results:
    #     plt.plot(new_results[t][q_test], new_results[t][j_pred], label=f"τ: {round(t,2)}", marker="o", linestyle="-")

    t = 0.2
    plt.plot(new_results[t][q_test], new_results[t][j_pred], label=f"τ: {round(t, 2)}", marker="o", linestyle="-")
    t = 0.4
    plt.plot(new_results[t][q_test], new_results[t][j_pred], label=f"τ: {round(t, 2)}", marker="o", linestyle="-")
    t = 0.8
    plt.plot(new_results[t][q_test], new_results[t][j_pred], label=f"τ: {round(t, 2)}", marker="o", linestyle="-")

    plt.xlabel("Q")
    plt.ylabel("Jc")
    plt.legend()
    plt.savefig(path_plots + "MUL/" + file)
    plt.show()
