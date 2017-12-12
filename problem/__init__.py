import copy
import operator as op
import numpy as np
import graphviz
import networkx as nx
from matplotlib import pyplot as plt
from string import ascii_lowercase
import itertools as it


class Problem(object):
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
        self.colors = colors

        triu_x, triu_y = np.triu_indices(self.n_nodes, 1)  # upper triangular coordinates
        self.n_connections = np.sum(self.connections[triu_x, triu_y])  # type: int

    def reset_colors(self):
        self.colors = np.repeat(self.colors[0], len(self.colors))

    @property
    def fitness(self):
        """
        The fitness of this graph - the number of edges that it's nodes
        doesn't have the same color, normalized to [0,1].

        :rtype: float
        :return: Ranges from 0 (worst fitness) to 1 (best fitness).
        """
        fitness = 0.

        for i in xrange(self.n_nodes):
            for j in xrange(i + 1, self.n_nodes):
                fitness += self.connections[i, j] * (self.colors[i] != self.colors[j])

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
            copy.copy(self.connections), copy.copy(self.colors)
        )
        return _copy

    def __deepcopy__(self, memo):
        _copy = self.__class__(
            copy.deepcopy(self.node_names), copy.deepcopy(self.available_colors),
            copy.deepcopy(self.connections), copy.deepcopy(self.colors)
        )

        return _copy

    def export(self, filename):
        raise NotImplementedError('not implemented yet!')

    @classmethod
    def generate_random(cls, n_nodes, available_colors):
        triu_x, triu_y = np.triu_indices(n_nodes, 1)  # upper triangular coordinates
        diag_x, diag_y = np.diag_indices(n_nodes)  # diagonal coordinates

        connections = np.empty((n_nodes, n_nodes), dtype=np.int32)
        connections[triu_x, triu_y] = np.random.randint(0, 2, size=len(triu_x), dtype=np.int32)
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

    def plot(self, show=False):
        plt.figure()
        G = nx.DiGraph()

        edges = zip(*np.where(self.connections))

        G.add_nodes_from(range(self.n_connections), label=self.node_names)
        G.add_edges_from(edges, label=self.node_names)

        layout = nx.circular_layout(G)
        nx.draw_networkx(G, pos=layout, node_color=self.colors)
        plt.axis('off')

        if show:
            plt.show()
