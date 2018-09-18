import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Graph:

    def __init__(self):
        self.node_count = 0
        self.factors = {}
        self.nodes = {}
        self.factors_count = 0

    def add_node(self, node_name,  alphabet_size):
        node_idx = self.node_count
        self.node_count += 1
        self.nodes[node_name] = [node_idx, alphabet_size, set()]

    def add_factor(self, factor_name, nodes, interaction_potential):
        for item in nodes:
            if item not in self.nodes:
                raise IndexError('Tried to factor non exciting node')
            self.nodes[item][2].add(factor_name)
        self.factors_count += 1
        self.factors[factor_name] = [nodes, np.exp(- interaction_potential)]

    def vis_graph(self):
        node_keys = self.nodes.keys()
        factor_keys = self.factors.keys()
        G = nx.Graph()
        G.add_nodes_from(node_keys)
        G.add_nodes_from(factor_keys)
        node_pos = {}
        factor_pos = {}
        pos = {}
        i = 0
        j = 0
        for item in self.nodes:
            temp = item
            node_pos[temp] = [i, j]
            pos[temp] = [i, j]
            i += 1
            for key in self.nodes[item][2]:
                G.add_edge(temp, key)
        i = 0
        j += 1
        for item in self.factors:
            factor_pos[item] = [i, j]
            pos[item] = [i, j]
            i += 1
        node_sub = G.subgraph(node_keys)
        factor_sub = G.subgraph(factor_keys)
        plt.figure()
        nx.draw_networkx(node_sub, pos=node_pos, node_color='b', node_shape='o', node_size=200)
        nx.draw_networkx(factor_sub, pos=factor_pos, node_color='r', node_shape='s')
        nx.draw_networkx_edges(G, pos=pos)
        plt.show()


def BP(Graph, t_max, epsilon):
    factors = Graph.factors
    nodes = Graph.nodes

    node2fac_messages = {}
    fac2node_messages = {}
    beliefs = {}
    delta = {}
    sum_of_alphabets = 0
    t_last = 0
    z = np.ones((t_max + 1), dtype=float)

    for A in factors:
        fac2node_messages[A] = {}
    for a in nodes:
        alphabet = nodes[a][1]
        sum_of_alphabets += alphabet
        node2fac_messages[a] = {}
        for A in nodes[a][2]:
            node2fac_messages[a][A] = np.ones(alphabet, dtype=float)/alphabet
        beliefs[a] = np.ones((alphabet, t_max + 1), dtype=float) / alphabet

    for t in range(t_max):
        t_last = t + 1
        counter = 0

        for A in factors:
            tensor_interaction = factors[A][1]
            node_neighbors = factors[A][0]
            factor_index_order = ''
            for object in node_neighbors:
                factor_index_order += object

            if len(node_neighbors) > 1:
                for a in node_neighbors:
                    for b in node_neighbors:
                        if a == b:
                            continue
                        else:
                            einsum_order = factor_index_order + ',' + b + '->' + factor_index_order
                            tensor_interaction = np.einsum(einsum_order, tensor_interaction, node2fac_messages[b][A])
                    einsum_order = factor_index_order + '->' + a
                    fac2node_messages[A][a] = np.einsum(einsum_order, tensor_interaction)
                    fac2node_messages[A][a] /= sum(fac2node_messages[A][a])
            else:
                fac2node_messages[A][node_neighbors[0]] = tensor_interaction
                fac2node_messages[A][node_neighbors[0]] /= sum(fac2node_messages[A][node_neighbors[0]])
        for a in nodes:
            for A in nodes[a][2]:
                if len(nodes[a][2]) > 1:
                    for B in nodes[a][2]:
                        if B == A:
                            continue
                        else:
                            node2fac_messages[a][A] = np.multiply(node2fac_messages[a][A], fac2node_messages[B][a])
                            beliefs[a][:, t + 1] = np.multiply(beliefs[a][:, t + 1], fac2node_messages[B][a])
                    beliefs[a][:, t + 1] = np.multiply(beliefs[a][:, t + 1], fac2node_messages[A][a])
                    beliefs[a][:, t + 1] /= sum(beliefs[a][:, t + 1])
                    delta[a] = abs(beliefs[a][:, t + 1] - beliefs[a][:, t])
                    node2fac_messages[a][A] /= sum(node2fac_messages[a][A])
                else:
                    continue

        for A in factors:
            boltzmann_factor = factors[A][1]
            for i in range(len(np.shape(boltzmann_factor))):
                boltzmann_factor = np.sum(boltzmann_factor, 0)
            z[t + 1] *= boltzmann_factor

        for a in nodes:
            for i in range(nodes[a][1]):
                if delta[a][i] <= epsilon:
                    counter += 1
        if counter == sum_of_alphabets:
            break

    return [beliefs, z, t_last]


def exact_partition(graph):
    factors = graph.factors
    z = 1
    for A in factors:
        boltzmann_factor = factors[A][1]
        for i in range(len(np.shape(boltzmann_factor))):
            boltzmann_factor = np.sum(boltzmann_factor, 0)
        z *= boltzmann_factor
    return z




''' check that messages are normalized !!!             - done
    what about np.exp() for the factors !!!!!!!!!      - done
    check factors are normelized                       - don't need to be
    check order of factors and neighbors of factors    - done 
    calculate beliefs
    check beliefs against simple system
    generalize for iterations up to t_max              - done
    add an exit with epsilon                           - done
    add exact marginal
'''

'''
    Remarks: 
            add_node('single small letter', size of alphabet)
            add_factor('single capital letter', array of neighbor nodes = ['a', 'c', 'd', ...], array of factor)
            
            factor: the factor should be in shape of neighbors alphabets.
            
            example: add_node('a', 2)
                     add_node('b', 3)
                     add_node('c', 4)
                     add_factor('A', np.array(['a', 'b', 'c']), np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
                                                                          [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]]))
                         
'''
'''
h = 1
k = 2
g = Graph()
g.add_node('a', 2)
g.add_node('b', 2)
g.add_node('c', 3)
g.add_node('d', 3)


g.add_factor('A', np.array(['a', 'b']), np.array([[1, 2], [5, 6]]))
g.add_factor('B', np.array(['a', 'b', 'c']), np.array([[[1, 2, 3], [4, 5, 6]], [[1, 1, 1], [2, 3, 6]]]))
g.add_factor('C', np.array(['a']), np.array([h, - h]))
g.add_factor('D', np.array(['b']), np.array([h, - h]))
g.add_factor('E', np.array(['a']), np.array([h, - h]))
g.add_factor('F', np.array(['c', 'd']), np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
g.add_factor('G', np.array(['b', 'd']), np.array([[0, 0, 1], [2, 0, 2]]))
g.vis_graph()

t_max = 100
[beliefs, z, t_last] = BP(g, t_max, 1e-20)
z_exact = exact_partition(g)

plt.figure()
plt.plot(range(t_last + 1), beliefs['a'][0, 0:(t_last + 1)], 'bo')
plt.plot(range(t_last + 1), beliefs['a'][1, 0:(t_last + 1)], 'ro')
plt.show()

plt.figure()
plt.plot(range(t_last + 1), z[0:(t_last + 1)], 'go')
plt.plot(z_exact, 'ro')
plt.show()
'''