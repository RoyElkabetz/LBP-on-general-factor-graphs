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
vis = 'no_grid'
n = 4
N = 0
N_of_sons = 2

for i in range(n):
    N += N_of_sons ** i
alpha = 2

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

# g.vis_graph(vis)
z = g.exact_partition()
F = - np.log(z)
beliefs, factor_beliefs = g.sum_product(t_max, 1)
beliefs_from_factor_beliefs = []
beliefs_from_factors_to_energy = np.zeros([g.node_count, t_max + 1, 2], dtype=float)
f_mean_field = np.ones(t_max + 1, dtype=float)
f_mean_field_from_factor_beliefs = np.ones(t_max + 1, dtype=float)
f_bethe = np.ones(t_max + 1, dtype=float)


# Initialization of single node beliefs calculated from factor beliefs
for i in range(g.node_count):
    beliefs_from_factor_beliefs.append({})
    for item in g.nodes[i][2]:
        beliefs_from_factor_beliefs[i][item] = np.ones([t_max + 1, 2])

# Calculating single node beliefs from factor beliefs
for t in range(t_max + 1):
    for i in range(g.node_count):
        for item in g.nodes[i][2]:
            beliefs_from_factor_beliefs[i][item][t] *= (np.einsum(factor_beliefs[item][t], range(g.node_count), [i]) / np.sum(factor_beliefs[item][t]))
        beliefs_from_factors_to_energy[i, t] += beliefs_from_factor_beliefs[i][item][t]

# Calculating free energies
for t in range(t_max + 1):
    factor_beliefs_for_F = {}
    for item in factor_beliefs:
        factor_beliefs_for_F[item] = factor_beliefs[item][t]
    f_mean_field[t] = g.mean_field_approx_to_F(beliefs[:, t])
    f_mean_field_from_factor_beliefs[t] = g.mean_field_approx_to_F(beliefs_from_factors_to_energy[:, t][:])
    f_bethe[t] = g.bethe_approx_to_F(beliefs[:, t], factor_beliefs_for_F)



# Plotting Data
plt.figure()
plt.title('Single node marginals')
for i in range(g.node_count):
    plt.plot(range(t_max + 1), beliefs[i], 'o')
plt.show()

plt.figure()
plt.title('Single node marginals calculated from factor beliefs')
#label = []
for i in range(g.node_count):
    for item in g.nodes[i][2]:
        #label.append(item)
        #label.append(item)
        plt.plot(range(t_max + 1), beliefs_from_factor_beliefs[i][item][:, 0], 'o')
        plt.plot(range(t_max + 1), beliefs_from_factor_beliefs[i][item][:, 1], 'o')
#plt.legend(label)
plt.show()

'''
j = 0
object = 'A'
plt.figure()
plt.title('comparing node marginals of a')
plt.plot(range(t_max), beliefs[j][0:t_max], 's')
plt.plot(range(t_max), beliefs_from_factor_beliefs[j][object][:, 0], 'o')
plt.plot(range(t_max), beliefs_from_factor_beliefs[j][object][:, 1], 'o')
#plt.plot(range(t_max), beliefs_from_factors_to_energy[j][0:t_max, 0], 's')
#plt.plot(range(t_max), beliefs_from_factors_to_energy[j][0:t_max, 1], 's')
#plt.legend(['a(1)', 'a(-1)', 'a_ha(1)', 'a_ha(-1)', '1', '-1'])
#plt.show()

delta0 = np.abs(beliefs[j][0:t_max, 0] - beliefs_from_factor_beliefs[j][object][:, 0])
delta1 = np.abs(beliefs[j][0:t_max, 1] - beliefs_from_factor_beliefs[j][object][:, 1])


plt.figure()
plt.title('error between marginal calculation over node a')
plt.plot(range(t_max), delta0, 'o')
plt.plot(range(t_max), delta1, 'o')
plt.show()
'''

plt.figure()
plt.title('Free energies')
plt.plot(range(t_max + 1), f_mean_field, 's')
plt.plot(range(t_max + 1), f_mean_field_from_factor_beliefs, 's')
plt.plot(range(t_max + 1), f_bethe, 'o')
plt.plot(range(t_max + 1), np.ones(t_max + 1, dtype=float) * F)
plt.legend(['F mean field', 'f_mean_field_from_factor_beliefs', 'F Bethe', 'F exact'])
plt.show()


'''
check if the marginals of single nodes from factor beliefs match the marginals from node belief - done
log base two for entropy
'''

