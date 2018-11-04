import numpy as np
import LBP_FactorGraphs as lbp
import Calculations_and_Plots as cplot


'''
    1D Ising model
'''

# parameters
h = 0.1
k = 1
t_max = 50
epsilon = 1e-7
N = 3
alpha = 2

# flags
vis = 'no vis'  # grid, no grid, no vis
single_node = 1
single_node_from_factors = 1
compare = 1
free_energies = 1
joint_flag = 1


# name of graph
g = lbp.Graph(N)

# nodes
for i in range(N):
    g.add_node(alpha)

# interactions
for i in range(N - 1):
    g.add_factor(np.array([i, i + 1]), np.array([[k, - k], [- k, k]]))
#g.add_factor(np.array([N - 1, 0]), np.array([[k, - k], [- k, k]]))  # periodic BC


# external field
for i in range(N):
    g.add_factor(np.array([i]), np.array([h, - h]))

# Implementing the algorithm
last_t = cplot.calc_n_plot(g, t_max, epsilon, vis, single_node, single_node_from_factors,  compare, free_energies, joint_flag)
