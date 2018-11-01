import numpy as np
import LBP_FactorGraphs_replica as lbp
import Calculations_and_Plots as cplot



'''
    Tree model
'''

# parameters
h = 0.1
k = 1
t_max = 30
n = 4
N = 0
N_of_sons = 2
for i in range(n):
    N += N_of_sons ** i
alpha = 2

# flags
vis = 'no vis'  # grid, no grid, no vis
single_node = 1
single_node_from_factors = 1
compare = 0
free_energies = 1
joint_flag = 1

# name of graph
g = lbp.Graph(N)
g.add_node(alpha)


# interactions
def append_children(graph, root_node, number_of_childrens, remaining_depth):
    if remaining_depth == 0:
        return 'done'
    for c in range(number_of_childrens):
        child = graph.add_node(alpha) - 1
        graph.add_factor(np.array([root_node, child]), np.array([[k, - k], [- k, k]]))
        append_children(graph, child, number_of_childrens, remaining_depth - 1)


append_children(g, 0, N_of_sons, n - 1)

# external field
for i in range(N):
    g.add_factor(np.array([i]), np.array([h, - h]))

# Implementing the algorithm
cplot.calc_n_plot(g, t_max, vis, single_node, single_node_from_factors,  compare, free_energies, joint_flag)
