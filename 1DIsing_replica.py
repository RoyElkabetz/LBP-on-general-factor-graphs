import numpy as np
import matplotlib.pyplot as plt
import LBP_FactorGraphs_replica as lbp
'''
    1D Ising model with cyclic BC 
'''

# parameters
h = 0.1
k = 1
t_max = 4

# name of graph
g = lbp.Graph()

# nodes
g.add_node('a', 2)
g.add_node('b', 2)
g.add_node('c', 2)
#g.add_node('d', 2)

# interactions
g.add_factor('A', np.array([0, 1]), np.array([[k, - k], [- k, k]]))
g.add_factor('B', np.array([1, 2]), np.array([[k, - k], [- k, k]]))
g.add_factor('C', np.array([2, 0]), np.array([[k, - k], [- k, k]]))
#g.add_factor('C', np.array(['c', 'd']), np.array([[k, - k], [- k, k]]))
#g.add_factor('D', np.array(['d', 'a']), np.array([[k, - k], [- k, k]]))

# external field
g.add_factor('E', np.array([0]), np.array([h, - h]))
g.add_factor('F', np.array([1]), np.array([h, - h]))
g.add_factor('G', np.array([2]), np.array([h, - h]))
#g.add_factor('H', np.array(['d']), np.array([h, - h]))


z = g.exact_partition()
F = - np.log(z)
beliefs = g.sum_product(t_max, 1)
node_beliefs = []
for i in range(g.node_count):
    node_beliefs.append(beliefs[i][t_max])
F_mean_field = g.mean_field_approx_to_F(node_beliefs)
z_mean_field = np.exp(- F_mean_field)


plt.figure()
plt.plot(range(t_max + 1), beliefs[0], 'o')
plt.plot(range(t_max + 1), beliefs[1], 'o')
plt.plot(range(t_max + 1), beliefs[2], 'o')
plt.show()