import numpy as np
from src.utils import *

# Values for the tests
step = 0.1  # Step for the values
max_j = 100  # Maximum value for the risk perception
js = np.arange(0, max_j, step)  # Risk perception
ts = np.arange(0, 1 + step, step)  # Tau values
qs = np.arange(0, 1 + step, step)  # Connection probability for Information Graph

iterations = 100  # Iteration
zero_threshold = 1e-4  # Zero threshold
eps = 1e-4  # Epsilon

# Values for the graph
n_nodes = 10000  # Number of nodes
k = 6  # Number of edges to attach from a new node
prob_k = k / n_nodes  # Probability for random graph

# Values for the infection
init_infect = 2  # Initial infected nodes
perc_init_infect = init_infect / n_nodes  # Initial percentage of infected nodes

immunity = healthy  # Immunity
rec_prob = 0.1  # Recovery probability
