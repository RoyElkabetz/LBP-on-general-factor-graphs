import numpy as np
import matplotlib.pyplot as plt
import LBP_FactorGraphs_replica as lbp
import copy as cp

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
g.add_node('a', 2)
g.add_node('b', 2)
g.add_node('c', 2)
#g.add_node('d', 2)
#g.add_node('e', 2)



# interactions
g.add_factor('A', np.array([0, 1]), np.array([[k, - k], [- k, k]]))
g.add_factor('B', np.array([1, 2]), np.array([[k, - k], [- k, k]]))
g.add_factor('C', np.array([2, 0]), np.array([[k, - k], [- k, k]]))
#g.add_factor('D', np.array([3, 4]), np.array([[k, - k], [- k, k]]))
#g.add_factor('E', np.array([4, 0]), np.array([[k, - k], [- k, k]]))



# external field
g.add_factor('ha', np.array([0]), np.array([h, - h]))
g.add_factor('hb', np.array([1]), np.array([h, - h]))
g.add_factor('hc', np.array([2]), np.array([h, - h]))
#g.add_factor('hd', np.array([3]), np.array([h, - h - 0.4]))
#g.add_factor('he', np.array([4]), np.array([h, - h - 0.5]))


# Running the algorithm
g.vis_graph()
z = g.exact_partition()
F = - np.log(z)
beliefs, factor_beliefs = g.sum_product(t_max, 1)
beliefs = np.array(beliefs)
beliefs_from_factor_beliefs = []
beliefs_from_factors_to_energy = np.zeros([g.node_count, t_max + 1, 2], dtype=float)
f_mean_field = np.ones(t_max, dtype=float)
f_mean_field_from_factor_beliefs = np.ones(t_max, dtype=float)
f_bethe = np.ones(t_max, dtype=float)
energy_per_factor_mf = np.ones([t_max, 6], dtype=float)
energy_per_factor_Bethe = np.ones([t_max, 6], dtype=float)



# Initialization of single node beliefs calculated from factor beliefs
for i in range(g.node_count):
    beliefs_from_factor_beliefs.append({})
    for item in g.nodes[i][2]:
        beliefs_from_factor_beliefs[i][item] = np.ones([t_max, 2])

# Calculating single node beliefs from factor beliefs
for t in range(t_max):
    for i in range(g.node_count):
        for item in g.nodes[i][2]:
            normalization = factor_beliefs[item][t]
            neighbors_of_item = cp.copy(list(g.factors[item][0]))
            new_shape = []
            for l in range(len(neighbors_of_item)):
                new_shape.append(g.nodes[neighbors_of_item[l]][1])
            normalization = np.reshape(normalization, new_shape)
            normalization = np.einsum(normalization, neighbors_of_item, [i])
            belief_of_i_from_item_at_t = cp.copy(normalization)
            normalization = np.reshape(np.sum(normalization, axis=0), [1])
            beliefs_from_factor_beliefs[i][item][t] *= (belief_of_i_from_item_at_t / normalization)
        beliefs_from_factors_to_energy[i, t] += belief_of_i_from_item_at_t / normalization

# Calculating free energies
for t in range(t_max):
    factor_beliefs_for_F = {}
    for item in factor_beliefs:
        factor_beliefs_for_F[item] = factor_beliefs[item][t]
    f_mean_field[t], energy_per_factor_mf[t] = g.mean_field_approx_to_F(beliefs[:, t])
    #f_mean_field_from_factor_beliefs[t] = g.mean_field_approx_to_F(beliefs_from_factors_to_energy[:, t])
    f_bethe[t], energy_per_factor_Bethe[t] = g.bethe_approx_to_F(beliefs[:, t], factor_beliefs_for_F)


# Plotting Data
plt.figure()
plt.title('Single node marginals')
for i in range(g.node_count):
    plt.plot(range(t_max + 1), beliefs[i], 'o')
plt.show()

plt.figure()
plt.title('Single node marginals calculated from factor beliefs')
label = []
for i in range(g.node_count):
    for item in g.nodes[i][2]:
        label.append(item)
        label.append(item)
        plt.plot(range(t_max), beliefs_from_factor_beliefs[i][item][:, 0], 'o')
        plt.plot(range(t_max), beliefs_from_factor_beliefs[i][item][:, 1], 'o')
plt.legend(label)
plt.show()

j = 0
object = 'A'
plt.figure()
plt.title('comparing node marginals of a')
plt.plot(range(t_max), beliefs[j][0:t_max], 's')
plt.plot(range(t_max), beliefs_from_factor_beliefs[j][object][:, 0], 'o')
plt.plot(range(t_max), beliefs_from_factor_beliefs[j][object][:, 1], 'o')
plt.plot(range(t_max), beliefs_from_factors_to_energy[j][0:t_max, 0], 's')
plt.plot(range(t_max), beliefs_from_factors_to_energy[j][0:t_max, 1], 's')
plt.legend(['a(1)', 'a(-1)', 'a_ha(1)', 'a_ha(-1)', '1', '-1'])
plt.show()

delta0 = np.abs(beliefs[j][0:t_max, 0] - beliefs_from_factor_beliefs[j][object][:, 0])
delta1 = np.abs(beliefs[j][0:t_max, 1] - beliefs_from_factor_beliefs[j][object][:, 1])


plt.figure()
plt.title('error between marginal calculation over node a')
plt.plot(range(t_max), delta0, 'o')
plt.plot(range(t_max), delta1, 'o')
plt.show()




plt.figure()
plt.title('Free energies')
plt.plot(range(t_max), f_mean_field, 's')
#plt.plot(range(t_max), f_mean_field_from_factor_beliefs, 's')
plt.plot(range(t_max), f_bethe, 'o')
plt.plot(range(t_max), np.ones(t_max, dtype=float) * F)
plt.legend(['F mean field', 'F Bethe', 'F exact'])
plt.show()

w = 3
plt.figure()
plt.title('energy per factor, MF and Bethe')
plt.plot(range(t_max), energy_per_factor_Bethe[:, w], 'o')
plt.plot(range(t_max), energy_per_factor_mf[:, w])
plt.plot(range(t_max), energy_per_factor_Bethe[:, w - 1], 'o')
plt.plot(range(t_max), energy_per_factor_mf[:, w - 1])
plt.show()
