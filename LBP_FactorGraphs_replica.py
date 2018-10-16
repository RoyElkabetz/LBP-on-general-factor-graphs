import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy as cp


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
        #print('complex1')
        #z = np.ones(np.array(alphabet), dtype=complex)
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
        node_belief = []
        factor_beliefs = {}

        '''
            Initialization of messages and beliefs
        '''
        for item in factors:
            factor_beliefs[item] = []

        for i in range(self.node_count):
            alphabet = nodes[i][1]
            node2factor.append({})
            node_belief.append([])
            for item in nodes[i][2]:
                node2factor[i][item] = np.ones(alphabet) / alphabet
                node2factor[i][item] = self.broadcasting(node2factor[i][item], np.array([i]))
            node_belief[i].append(np.ones(alphabet) / alphabet)

        '''
            Preforming sum product iterations
        '''
        for t in range(t_max):
            for item in factors:
                neighbors_nodes = factors[item][0]
                summing_order = np.flip(np.sort(cp.copy(neighbors_nodes)), axis=0)
                factor2node[item] = {}
                temp_factor = cp.deepcopy(factors[item][1])
                for i in range(len(neighbors_nodes)):
                    vec = range(self.node_count)
                    temp = cp.deepcopy(factors[item][1])  # temp is holding the message until it is ready
                    for j in range(len(neighbors_nodes)):
                        if neighbors_nodes[j] == neighbors_nodes[i]:
                            continue
                        else:
                            vec.remove(neighbors_nodes[j])
                            temp *= node2factor[neighbors_nodes[j]][item]

                    temp_factor = temp_factor * node2factor[neighbors_nodes[i]][item]  # temp_factor is holding the factor belief until it is ready
                    temp_normalization = cp.copy(temp_factor)
                    for n in range(len(summing_order)):
                        temp_normalization = np.sum(temp_normalization, axis=summing_order[n])

                    temp_normalization = np.reshape(temp_normalization, [1])
                    temp = np.einsum(temp, range(self.node_count), vec)
                    vec2 = cp.copy(vec)
                    vec2.remove(neighbors_nodes[i])
                    factor2node[item][neighbors_nodes[i]] = np.reshape(temp / np.einsum(temp, vec, vec2), nodes[neighbors_nodes[i]][1])
                factor_beliefs[item].append(np.array(temp_factor / temp_normalization))

            for i in range(self.node_count):
                alphabet = nodes[i][1]
                neighbors_factors = nodes[i][2]
                temp = 1
                for item in neighbors_factors:
                    node2factor[i][item] = np.ones(alphabet)
                    node2factor[i][item] = self.broadcasting(node2factor[i][item], np.array([i]))
                    for object in neighbors_factors:
                        if object == item:
                            continue
                        else:
                            node2factor[i][item] *= self.broadcasting(np.array(factor2node[object][i]), np.array([i]))

                    node2factor[i][item] /= np.sum(node2factor[i][item], axis=i)
                    temp *= factor2node[item][i]

                node_belief[i].append(temp / np.sum(temp, axis=0))

        return node_belief, factor_beliefs

    def mean_field_approx_to_F(self, node_beliefs):
        energy = 0
        entropy = 0
        energy_per_factor = np.ones([1, 6], dtype=float)
        s = 0
        for item in self.factors:
            temp_energy = cp.deepcopy(self.factors[item][1])
            neighbors = cp.deepcopy(self.factors[item][0])
            temp_energy = - np.log(temp_energy)
            summing_vec = range(self.node_count)
            for i in range(len(neighbors)):
                temp_energy *= self.broadcasting(node_beliefs[neighbors[i]], np.array([neighbors[i]]))
                summing_vec.remove(neighbors[i])
            energy_per_factor[0, s] *= np.reshape(np.einsum(temp_energy, range(self.node_count), summing_vec), [1])
            s += 1
            energy += np.reshape(np.einsum(temp_energy, range(self.node_count), summing_vec), [1])
        for i in range(self.node_count):
            entropy -= np.dot(node_beliefs[i], np.log(node_beliefs[i]))
        F_approx = energy - entropy
        print('H_MF', entropy)
        print('U_MF', energy)
        return F_approx , energy_per_factor

    def bethe_approx_to_F(self, node_beliefs, factor_beliefs):
        energy = 0
        entropy = 0
        energy_per_factor = np.ones([1, len(self.factors.keys())], dtype=float)
        s = 0
        for item in self.factors:
            temp_energy = cp.deepcopy(self.factors[item][1])
            neighbors = cp.deepcopy(self.factors[item][0])
            summing_vec = range(self.node_count)
            for i in range(len(neighbors)):
                summing_vec.remove(neighbors[i])
            energy += np.reshape(np.einsum(- factor_beliefs[item] * np.log(temp_energy), range(self.node_count), summing_vec), [1])
            entropy += np.reshape(np.einsum(- factor_beliefs[item] * np.log(factor_beliefs[item]), range(self.node_count), summing_vec), [1])
            energy_per_factor[0, s] *= np.reshape(np.einsum(- factor_beliefs[item] * np.log(temp_energy), range(self.node_count), summing_vec), [1])
            s += 1
        for i in range(self.node_count):
            d = len(self.nodes[i][2])
            entropy += (d - 1) * np.dot(node_beliefs[i], np.log(node_beliefs[i]))
          #  print(d)
         #   print((d - 1) * np.dot(node_beliefs[i], np.log(node_beliefs[i])))
        F_bethe_approx = energy - entropy
        print('H_Bethe', entropy)
        print('U_Bethe', energy)
        return F_bethe_approx, energy_per_factor

    def vis_graph(self):
        node_keys = []
        for i in range(self.node_count):
            node_keys.append(self.nodes[i][0])
        factor_keys = self.factors.keys()
        G = nx.Graph()
        G.add_nodes_from(node_keys)
        G.add_nodes_from(factor_keys)
        node_pos = {}
        factor_pos = {}
        pos = {}
        i = 0
        j = 0
        for item in range(self.node_count):
            temp = cp.copy(self.nodes[item][0])
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
