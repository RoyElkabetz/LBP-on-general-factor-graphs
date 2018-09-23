import numpy as np
import matplotlib.pyplot as plt
import LBP_FactorGraphs as lbp
'''
    1D Ising model with cyclic BC 
'''

# parameters
h = 0.1
k = 1

# name of graph
g = lbp.Graph()

# nodes
g.add_node('a', 2)
g.add_node('b', 2)
g.add_node('c', 2)
#g.add_node('d', 2)

# interactions
g.add_factor('A', np.array(['a', 'b']), np.array([[k, - k], [- k, k]]))
g.add_factor('B', np.array(['b', 'c']), np.array([[k, - k], [- k, k]]))
g.add_factor('C', np.array(['c', 'a']), np.array([[k, - k], [- k, k]]))
#g.add_factor('C', np.array(['c', 'd']), np.array([[k, - k], [- k, k]]))
#g.add_factor('D', np.array(['d', 'a']), np.array([[k, - k], [- k, k]]))

# external field
g.add_factor('E', np.array(['a']), np.array([h, - h]))
g.add_factor('F', np.array(['b']), np.array([h, - h]))
g.add_factor('G', np.array(['c']), np.array([h, - h]))
#g.add_factor('H', np.array(['d']), np.array([h, - h]))


g.vis_graph()

t_max = 100
[beliefs, z, t_last] = lbp.BP(g, t_max, 1e-5)
z_exact = lbp.exact_partition(g)

plt.figure()
plt.plot(range(t_last + 1), beliefs['a'][0, 0:(t_last + 1)], 'bo')
plt.plot(range(t_last + 1), beliefs['a'][1, 0:(t_last + 1)], 'ro')
plt.show()

plt.figure()
plt.plot(range(t_last + 1), z[0:(t_last + 1)], 'go')
plt.plot(z_exact, 'ro')
plt.show()
