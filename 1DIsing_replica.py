import numpy as np
import matplotlib.pyplot as plt
import LBP_FactorGraphs_replica as lbp
'''
    1D Ising model with cyclic BC 
'''

# parameters
h = 0.1
k = 1
t_max = 30

# name of graph
g = lbp.Graph()

# nodes
g.add_node(2)
g.add_node(2)
g.add_node(2)
g.add_node(2)
g.add_node(2)
g.add_node(2)
g.add_node(2)

# interactions

#g.add_factor('A', np.array([0, 1]), np.array([[k, - k], [- k, k]]))
#g.add_factor('B', np.array([1, 2]), np.array([[k, - k], [- k, k]]))
#g.add_factor('C', np.array([2, 3]), np.array([[k, - k], [- k, k]]))
g.add_factor(np.array([3, 4]), np.array([[k, - k], [- k, k]]))
#g.add_factor('E', np.array([4, 5]), np.array([[k, - k], [- k, k]]))
#g.add_factor('F', np.array([5, 6]), np.array([[k, - k], [- k, k]]))
#g.add_factor('G', np.array([6, 0]), np.array([[k, - k], [- k, k]]))

#g.add_factor('C', np.array([2, 0]), np.array([[k, - k], [- k, k]]))
#g.add_factor('C', np.array([2, 3]), np.array([[k, - k], [- k, k]]))
#g.add_factor('D', np.array([3, 0]), np.array([[k, - k], [- k, k]]))

# external field
g.add_factor(np.array([0]), np.array([h, - h]))
g.add_factor(np.array([1]), np.array([h, - h]))
g.add_factor(np.array([2]), np.array([h, - h]))
g.add_factor(np.array([3]), np.array([h, - h]))
g.add_factor(np.array([4]), np.array([h, - h]))
g.add_factor(np.array([5]), np.array([h, - h]))
g.add_factor(np.array([6]), np.array([h, - h]))

#g.add_factor('H', np.array([3]), np.array([h, - h]))

g.vis_graph()
z = g.exact_partition()
F = - np.log(z)
beliefs = g.sum_product(t_max, 1)
beliefs = np.array(beliefs)
f_mean_field = np.ones(t_max, dtype=float)
for t in range(t_max):
    f_mean_field[t] = g.mean_field_approx_to_F(beliefs[:, t])

plt.figure()
plt.title('Single node marginals')
for i in range(g.node_count):
    plt.plot(range(t_max + 1), beliefs[i], 'o')
plt.show()

plt.figure()
plt.title('Free energies')
plt.plot(range(t_max), f_mean_field, 'o')
plt.plot(range(t_max), np.ones(t_max, dtype=float) * F, 'o')
plt.show()
