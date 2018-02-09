import copy
import itertools as it
import operator as op
from string import ascii_lowercase

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from functools import reduce


class Graph(object):
    def plot(self, title=''):
        plt.figure()
        plt.axis('off')
        plt.title(title)


class Problem(Graph):
    def __init__(self, variable_names, available_values):
        self.available_values = available_values
        self.variable_names = variable_names
        self.n_variables = len(variable_names)

    @property
    def fitness(self):
        return None

    def __copy__(self):
        _copy = self.__class__(self.variable_names, self.available_values)
        return _copy

    def __deepcopy__(self, memo):
        variable_names = copy.deepcopy(self.variable_names)
        available_values = copy.deepcopy(self.available_values)

        _copy = self.__class__(variable_names, available_values)
        return _copy


class ModelGraph(Problem):
    def __init__(self, node_names, available_colors, connections, colors):
        """
        Constructor for MGraph class.

        :type connections: numpy.ndarray
        :param connections: A boolean matrix of size n_nodes x n_nodes, where 0 indicates no relation
            and 1 indicates a connection between the nodes.
        :type colors: list
        :param colors: a list where each entry is the color for that node
        :type available_colors: list
        :param available_colors: a list with all available colors for this problem.
        """
        super(ModelGraph, self).__init__(variable_names=node_names, available_values=available_colors)

        self.connections = connections
        self._colors = colors

        triu_x, triu_y = np.triu_indices(self.n_nodes, 1)  # upper triangular coordinates
        self.n_connections = np.sum(self.connections[triu_x, triu_y])  # type: int

    def reset_colors(self):
        self._colors = np.repeat(self._colors[0], len(self._colors))

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, new_colors):
        self._colors = list(new_colors)

    @property
    def fitness(self):
        """
        The fitness of this graph - the number of edges that it's nodes
        doesn't have the same color, normalized to [0,1].

        :rtype: float
        :return: Ranges from 0 (worst fitness) to 1 (best fitness).
        """
        fitness = 0.

        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                connected = self.connections[i, j]
                diff_color = (self._colors[i] != self._colors[j])
                fitness += connected * diff_color

        return fitness / float(self.n_connections)

    @property
    def available_colors(self):
        return self.available_values

    @property
    def node_names(self):
        return self.variable_names

    @property
    def n_nodes(self):
        return self.n_variables

    def __copy__(self):
        _copy = self.__class__(
            copy.copy(self.node_names), copy.copy(self.available_colors),
            copy.copy(self.connections), copy.copy(self._colors)
        )
        return _copy

    def __deepcopy__(self, memo):
        _copy = self.__class__(
            copy.deepcopy(self.node_names), copy.deepcopy(self.available_colors),
            copy.deepcopy(self.connections), copy.deepcopy(self._colors)
        )

        return _copy

    @classmethod
    def generate_random(cls, n_nodes, available_colors, prob=0.25):
        triu_x, triu_y = np.triu_indices(n_nodes, 1)  # upper triangular coordinates
        diag_x, diag_y = np.diag_indices(n_nodes)  # diagonal coordinates

        connections = np.empty((n_nodes, n_nodes), dtype=np.int32)
        connections[triu_x, triu_y] = np.random.choice([0, 1], replace=True, size=len(triu_x), p=[1. - prob, prob]).astype(np.int32)
        connections[triu_y, triu_x] = connections[triu_x, triu_y]
        connections[diag_x, diag_y] = False

        picked_colors = np.random.choice(available_colors, n_nodes, replace=True)

        node_names = []
        pool = list(ascii_lowercase)
        while len(node_names) < n_nodes:
            node_names = (node_names + pool)[:n_nodes]
            pool = map(
                lambda x: reduce(
                    op.add,
                    x
                ),
                it.product(pool, ascii_lowercase)
            )

        return cls(
            node_names=node_names,
            available_colors=available_colors,
            connections=connections,
            colors=picked_colors
        )

    def plot(self, title=''):
        super(ModelGraph, self).plot(title)

        G = nx.Graph()

        edges = zip(*np.where(self.connections))

        edges = list(map(
            lambda x: list(map(
                lambda y: self.node_names[y],
                x
            )),
            edges
        ))

        G.add_nodes_from(self.node_names)
        G.add_edges_from(edges)

        dict_conv = dict(
            zip(self.node_names, range(self.n_nodes))
        )

        __colors = [self._colors[dict_conv[x]] for x in G.nodes()]

        nx.draw_networkx(G, node_color=__colors)


class WorldMap(Problem):
    def __init__(self, variable_names, available_values):
        super().__init__(variable_names, available_values)
