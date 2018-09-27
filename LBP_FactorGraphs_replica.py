import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Graph:

    def __init__(self):
        self.node_count = 0
        self.factors = {}
        self.nodes = []
        self.factors_count = 0

    def add_node(self, node_name, alphabet_size):
        self.node_count += 1
        self.nodes.append([node_name, alphabet_size, set()])

    def broadcasting(self, factor, nodes):
        new_shape = np.ones(self.node_count, dtype=np.int)
        new_shape[nodes] = np.shape(factor)
        return np.reshape(factor, new_shape)

    def add_factor(self, factor_name, factor_nodes, boltzmann_factor):
        for i in range(len(factor_nodes)):
            if factor_nodes[i] > self.node_count:
                raise IndexError('Tried to factor non exciting node')
            self.nodes[factor_nodes[i]][2].add(factor_name)
        self.factors_count += 1
        self.factors[factor_name] = [factor_nodes, self.broadcasting(np.exp(- boltzmann_factor), factor_nodes)]

    def exact_partition(self):
        alphabet = np.zeros(self.node_count, dtype=np.int)
        for i in range(self.node_count):
            alphabet[i] = self.nodes[i][1]
        z = np.ones(np.array(alphabet), dtype=float)
        for item in self.factors:
            z *= self.factors[item][1]
        for i in range(self.node_count):
            z = np.sum(z, 0)
        return z

    def sum_product(self, t_max, epsilon):
        factors = self.factors
        nodes = self.nodes
        node2factor = []
        factor2node = {}
        for i in range(self.node_count):
            alphabet = nodes[i][1]
            node2factor.append(np.ones(alphabet) / alphabet)
            node2factor[i] = self.broadcasting(node2factor[i], i)
        for t in range(t_max):
            for item in factors:
                neighbors_nodes = factors[item][0]
                factor2node[item] = []
                temp = factors[item][1]
                for i in range(len(neighbors_nodes)):
                    for j in range(len(neighbors_nodes)):
                        if neighbors_nodes[j] == neighbors_nodes[i]:
                            continue
                        else:
                            temp *= node2factor[neighbors_nodes[j]]
                    for j in range(len(neighbors_nodes)):
                        if neighbors_nodes[j] == neighbors_nodes[i]:
                            continue
                        else:
                            temp = np.sum(temp, axis=neighbors_nodes[j])
                    factor2node[item].append([neighbors_nodes[i], temp / np.sum(temp, axis=neighbors_nodes[i])])
            for i in range(self.node_count):



'''
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
'''

h = 1
k = 2
g = Graph()
g.add_node('a', 2)
g.add_node('b', 2)
g.add_node('c', 3)
g.add_node('d', 3)


g.add_factor('A', np.array([0, 1]), np.array([[1, 2], [5, 6]]))
g.add_factor('B', np.array([0, 1, 2]), np.array([[[1, 2, 3], [4, 5, 6]], [[1, 1, 1], [2, 3, 6]]]))
g.add_factor('C', np.array([0]), np.array([h, - h]))
g.add_factor('D', np.array([1]), np.array([h, - h]))
g.add_factor('E', np.array([0]), np.array([h, - h]))
g.add_factor('F', np.array([2, 3]), np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
g.add_factor('G', np.array([1, 3]), np.array([[0, 0, 1], [2, 0, 2]]))

z = g.exact_partition()
