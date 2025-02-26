import warnings

from src.graphs import *
from src.plot import plot_all_graphs, plot_critical_t, plot_comparison_j_q, plot_critical_j, \
    plot_percolation_critical_j, plot_q_value
from src.save_csv import save_results, load_results, load_results_mul
from src.tests import *
from src.config import *


def view_graphs():
    # RANDOM GRAPH WITH K
    PG, VG = random_graph_test(n_nodes, pPG=prob_k, pVG=prob_k)
    IG = get_information_graph(PG, VG, q=0.5)
    plot_all_graphs(PG, VG, IG)

    # RANDOM GRAPH WITH K*2
    kk = 2 * k
    prob_kk = kk / n_nodes
    PG, VG = random_graph_test(n_nodes, pPG=prob_kk, pVG=prob_kk)
    IG = get_information_graph(PG, VG, q=0.5)
    plot_all_graphs(PG, VG, IG)

    # CYCLE GRAPH
    PG, VG = cycle_graph_test(n_nodes)
    IG = get_information_graph(PG, VG, q=0.5)
    plot_all_graphs(PG, VG, IG)

    # SCALE FREE GRAPH WITH K
    PG, VG = scale_free_graph_test(n_nodes, mPG=prob_k, mVG=prob_k)
    IG = get_information_graph(PG, VG, q=0.5)
    plot_all_graphs(PG, VG, IG)


def mean_field_test():
    test = "MF/"
    results = {}

    # RANDOM GRAPH WITH K = 6
    kk = 6
    graph_type = "<k>=" + str(kk)
    result = mean_field_jc_test(kk, perc_init_infect, iterations, ts, js)
    path_test = "k"+str(kk)+"-int"+str(iterations)
    save_results(result, file=test+path_test+".csv")
    results[graph_type] = result
    plot_critical_j({graph_type: result}, file=test+path_test+".png")

    # RANDOM GRAPH WITH K = 4
    kk = 4
    graph_type = "<k>=" + str(kk)
    result = mean_field_jc_test(kk, perc_init_infect, iterations, ts, js)
    path_test = "k"+str(kk)+"-int"+str(iterations)
    save_results(result, file=test+path_test+".csv")
    results[graph_type] = result
    plot_critical_j({graph_type: result}, file=test+path_test+".png")

    # RANDOM GRAPH WITH K = 2
    kk = 2
    graph_type = "<k>=" + str(kk)
    result = mean_field_jc_test(kk, perc_init_infect, iterations, ts, js)
    path_test = "k"+str(kk)+"-int"+str(iterations)
    save_results(result, file=test+path_test+".csv")
    results[graph_type] = result
    plot_critical_j({graph_type: result}, file=test+path_test+".png")

    # PLOT ALL RESULTS
    plot_critical_j(results, file=test+mean_field_jc_plot+"-"+str(iterations)+".png")


def simple_percolation_test():
    test = "SIM/"
    results = {}

    # RANDOM GRAPH WITH K
    PG, VG = random_graph_test(n_nodes, pPG=prob_k, pVG=prob_k)
    graph_type = "Poisson <k>=" + str(k)
    result = simple_tau_percolation_test(PG, iterations, ts)
    path_test = "Poisson-k"+str(k)+"_nodes"+str(n_nodes)+"_it"+str(iterations)
    save_results(result, file=test+path_test+".csv")
    results[graph_type] = result
    plot_critical_t({graph_type: result}, file=test+path_test+".png")

    # RANDOM GRAPH WITH K*2
    kk = 2 * k
    prob_kk = kk / n_nodes
    PG, VG = random_graph_test(n_nodes, pPG=prob_kk, pVG=prob_kk)
    graph_type = "Poisson <k>=" + str(kk)
    result = simple_tau_percolation_test(PG, iterations, ts)
    path_test = "Poisson-k"+str(k)+"_nodes"+str(n_nodes)+"_it"+str(iterations)
    save_results(result, file=test+path_test+".csv")
    results[graph_type] = result
    plot_critical_t({graph_type: result}, file=test+path_test+".png")

    # CYCLE GRAPH
    PG, VG = cycle_graph_test(n_nodes)
    graph_type = "Cycle"
    result = simple_tau_percolation_test(PG, iterations, ts)
    path_test = "Cycle-nodes"+str(n_nodes)+"_it"+str(iterations)
    save_results(result, file=test+path_test+".csv")
    results[graph_type] = result
    plot_critical_t({graph_type: result}, file=test+path_test+".png")

    # SCALE FREE GRAPH WITH K
    PG, VG = scale_free_graph_test(n_nodes, mPG=k, mVG=k)
    graph_type = "Scale Free <k>=" + str(k)
    result = simple_tau_percolation_test(PG, iterations, ts)
    path_test = "ScaleFree-k"+str(k)+"_nodes"+str(n_nodes)+"_it"+str(iterations)
    save_results(result, file=test+path_test+".csv")
    results[graph_type] = result
    plot_critical_t({graph_type: result}, file=test+path_test+".png")

    # PLOT ALL RESULTS
    plot_critical_t(results, file=test + simple_tc_plot+"-nodes"+str(n_nodes)+"_it"+str(iterations)+".png")


def percolation_test():
    test = "PERC/"
    results = {}

    # RANDOM GRAPH WITH K
    PG, VG = random_graph_test(n_nodes, pPG=prob_k, pVG=prob_k)
    graph_type = "Poisson <k>=" + str(k)
    result = risk_percolation_j_test(PG, iterations, ts)
    path_test = "Poisson-k"+str(k)+"_nodes"+str(n_nodes)+"_it"+str(iterations)
    save_results(result, file=test+path_test+".csv")
    results[graph_type] = result
    plot_percolation_critical_j({graph_type: result}, file=test+path_test+".png")

    # CYCLE GRAPH
    PG, VG = cycle_graph_test(n_nodes)
    graph_type = "Cycle"
    result = risk_percolation_j_test(PG, iterations, ts)
    path_test = "Cycle-nodes"+str(n_nodes)+"_it"+str(iterations)
    save_results(result, file=test+path_test+".csv")
    results[graph_type] = result
    plot_percolation_critical_j({graph_type: result}, file=test+path_test+".png")

    # SCALE FREE GRAPH WITH K
    PG, VG = scale_free_graph_test(n_nodes, mPG=k, mVG=k)
    graph_type = "Scale Free <k>=" + str(k)
    result = risk_percolation_j_test(PG, iterations, ts)
    path_test = "ScaleFree-k"+str(k)+"_nodes"+str(n_nodes)+"_it"+str(iterations)
    save_results(result, file=test+path_test+".csv")
    results[graph_type] = result
    plot_percolation_critical_j({graph_type: result}, file=test+path_test+".png")

    # PLOT ALL RESULTS
    plot_percolation_critical_j(results, file=test + percolation_jc_plot + "-k"+str(k)+"_nodes"+str(n_nodes)+"_it"+str(iterations)+".png")


def multiplex_percolation_test():
    test = "MUL/"

    PG, VG = random_graph_test(n_nodes, pPG=prob_k, pVG=prob_k)
    # graph_type = "Poisson <k>=" + str(k)
    results = multiplex_percolation_jc_test(PG, VG, iterations, ts, qs)
    path_test = "Poisson-k"+str(k)+"_nodes"+str(n_nodes)+"_it"+str(iterations)+"_jm"+str(max_j)
    save_results(results, file=test+path_test+".csv")
    plot_q_value(results, file=test+path_test+".png")
    # TODO Visualize the comparison between j and q in the multiplex percolation test
    # plot_comparison_j_q(results, "comparisonjq"+path_test+".png")

    PG, VG = cycle_graph_test(n_nodes)
    # graph_type = "Cycle"
    results = multiplex_percolation_jc_test(PG, VG, iterations, ts, qs)
    path_test = "Cycle-nodes"+str(n_nodes)+"_it"+str(iterations)+"_jm"+str(max_j)
    save_results(results, file=test+path_test+".csv")
    plot_q_value(results, file=test+path_test+".png")
    # TODO Visualize the comparison between j and q in the multiplex percolation test
    # plot_comparison_j_q(results, "comparisonjq"+path_test+".png")

    PG, VG = scale_free_graph_test(n_nodes, mPG=k, mVG=k)
    # graph_type = "Scale Free <k>=" + str(k)
    results = multiplex_percolation_jc_test(PG, VG, iterations, ts, qs)
    path_test = "ScaleFree-k"+str(k)+"_nodes"+str(n_nodes)+"_it"+str(iterations)+"_jm"+str(max_j)
    save_results(results, file=test+path_test+".csv")
    plot_q_value(results, file=test+path_test+".png")
    # TODO Visualize the comparison between j and q in the multiplex percolation test
    # plot_comparison_j_q(results, "comparisonjq"+path_test+".png")

    PG, _ = random_graph_test(n_nodes, pPG=prob_k, pVG=prob_k)
    _, VG = scale_free_graph_test(n_nodes, mPG=k, mVG=k)
    # graph_type = "Poisson <k>=" + str(k) + " and Scale Free <k>=" + str(k)
    results = multiplex_percolation_jc_test(PG, VG, iterations, ts, qs)
    path_test = "Poisson-k"+str(k)+"+ScaleFree-k"+str(k)+"_nodes"+str(n_nodes)+"_it"+str(iterations)+"_jm"+str(max_j)
    save_results(results, file=test+path_test+".csv")
    plot_q_value(results, file=test+path_test+".png")
    # TODO Visualize the comparison between j and q in the multiplex percolation test
    # plot_comparison_j_q(results, "comparisonjq"+path_test+".png")


def visualize_comparison():
    # Load results con jminf
    """
    results = load_results_mul("MUL/Cycle-nodes10000_it100_jminf.csv")
    plot_comparison_j_q(results, "comparisonjq-Cycle-nodes10000_it100_jminf.png")

    results = load_results_mul("MUL/Poisson-k3_nodes10000_it100_jminf.csv")
    plot_comparison_j_q(results, "comparisonjq-Poisson-k3_nodes10000_it100_jminf.png")

    results = load_results_mul("MUL/ScaleFree-k3_nodes10000_it100_jminf.csv")
    plot_comparison_j_q(results, "comparisonjq-ScaleFree-k3_nodes10000_it100_jminf.png")

    results = load_results_mul("MUL/Poisson-k6+ScaleFree-k6_nodes10000_it100_jminf.csv")
    plot_comparison_j_q(results, "comparisonjq-Poisson-k6+ScaleFree-k6_nodes10000_it100_jminf.png")
    """

    # Load results con jm100

    results = load_results_mul("MUL/Cycle-nodes10000_it100_jm100.csv")
    plot_comparison_j_q(results, "zcjq-Cycle-nodes10000_it100_jm100.png")

    results = load_results_mul("MUL/Poisson-k6_nodes10000_it100_jm100.csv")
    plot_comparison_j_q(results, "zcjq-Poisson-k3_nodes10000_it100_jm100.png")

    results = load_results_mul("MUL/ScaleFree-k6_nodes10000_it100_jm100.csv")
    plot_comparison_j_q(results, "zcjq-ScaleFree-k3_nodes10000_it100_jm100.png")

    results = load_results_mul("MUL/Poisson-k6+ScaleFree-k6_nodes10000_it100_jm100.csv")
    plot_comparison_j_q(results, "zcjq-Poisson-k6+ScaleFree-k6_nodes10000_it100_jm100.png")


def main():
    # Ignore warnings
    warnings.filterwarnings("ignore")

    # GRAPH TESTS
    # test_graphs()

    # TESTS
    # mean_field_test()
    # simple_percolation_test()
    # percolation_test()
    # multiplex_percolation_test()
    visualize_comparison()


if __name__ == "__main__":
    main()
