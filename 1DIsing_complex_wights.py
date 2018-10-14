import numpy as np
import matplotlib.pyplot as plt
import LBP_FG_complex as lbp
'''
    1D Ising model with cyclic BC 
'''

# parameters
h = 0.1j
k = 1j
t_max = 30

# name of graph
g = lbp.Graph()

# nodes
g.add_node('a', 2)
g.add_node('b', 2)
g.add_node('c', 2)
g.add_node('d', 2)
g.add_node('e', 2)


# interactions
g.add_factor('A', np.array([0, 1]), np.array([[k, - k], [- k, k]]))
g.add_factor('B', np.array([1, 2]), np.array([[k, - k], [- k, k]]))
g.add_factor('C', np.array([2, 3]), np.array([[k, - k], [- k, k]]))
g.add_factor('D', np.array([3, 4]), np.array([[k, - k], [- k, k]]))
g.add_factor('E', np.array([4, 0]), np.array([[k, - k], [- k, k]]))



# external field
g.add_factor('ha', np.array([0]), np.array([h, - h]))
g.add_factor('hb', np.array([1]), np.array([h, - h]))
g.add_factor('hc', np.array([2]), np.array([h, - h]))
g.add_factor('hd', np.array([3]), np.array([h, - h]))
g.add_factor('he', np.array([4]), np.array([h, - h]))


g.vis_graph()
z = g.exact_partition()
F = - np.log(z)
beliefs, factor_beliefs = g.sum_product(t_max, 1)
beliefs = np.array(beliefs)

f_mean_field = np.ones(t_max, dtype=complex)
#f_bethe = np.ones(t_max, dtype=float)
for t in range(t_max):
    factor_beliefs_for_F = {}
   # for item in factor_beliefs:
    #    factor_beliefs_for_F[item] = factor_beliefs[item][t]
    f_mean_field[t] = g.mean_field_approx_to_F(beliefs[:, t])
    #f_bethe[t] = g.bethe_approx_to_F(beliefs[:, t], factor_beliefs_for_F)

plt.figure()
plt.title('Single node marginals')

for i in range(g.node_count):
    plt.plot(range(t_max + 1), np.abs(beliefs[i]), 'o')
plt.legend('')
plt.show()

plt.figure()
plt.title('Free energies')
plt.plot(range(t_max), np.abs(f_mean_field), 's')
#plt.plot(range(t_max), f_bethe, 'o')
plt.plot(range(t_max), np.ones(t_max, dtype=float) * np.abs(F))
plt.legend(['F mean field', 'F exact'])
plt.show()
