import numpy as np
import LBP_FactorGraphs_complex as lbp
import Calculations_and_Plots_complex as cplot


'''
    2D Ising model with cyclic BC and complex weights
'''

# parameters
h = 0.1j
k = 1j
t_max = 100
N = 16
L = int(np.sqrt(N))
alphabet = 2
grid = np.reshape([range(N)], [L, L])

# flags
vis = 'grid'   # grid, no grid, no vis
single_node = 1
single_node_from_factors = 1
compare = 1
free_energies = 1
joint_flag = 1



# name of graph
g = lbp.Graph(N)

# nodes
for i in range(N):
    g.add_node(alphabet)

# external fields
for i in range(N):
    g.add_factor(np.array([i]), np.array([h, - h]))

# interactions
for i in range(L):
    for j in range(L):
        g.add_factor(np.array([grid[i, j], grid[i, np.mod(j + 1, L)]]), np.array([[k, - k], [- k, k]]))
        g.add_factor(np.array([grid[i, j], grid[np.mod(i + 1, L), j]]), np.array([[k, - k], [- k, k]]))


# Implementing the algorithm
cplot.calc_n_plot(g, t_max, vis, single_node, single_node_from_factors,  compare, free_energies, joint_flag)
