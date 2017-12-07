import copy
import operator as op
import numpy as np
import graphviz
import networkx as nx
from matplotlib import pyplot as plt
from string import ascii_lowercase
import itertools as it

class ModelGraph(object):
    """
    Attributes:

    :type nodes: list
    :param nodes: a list of the nodes.

    :type edges: list
    :param edges: a list of the edges between the nodes.
    """
    nodes = dict()
    edges = []
    _fitness = 0.
    names = []

    def __init__(self, connections=None, **kwargs):
        """
        Constructor for MGraph class.

        :type connections: numpy.ndarray
        :param node: optional -- A boolean matrix of size n_nodes x n_nodes, where 0 indicates no relation
            and 1 indicates a connection between the nodes.

        :type visual: graphviz.graph
        :param visual: optional - visual engine of the graph

        :type edges: list
        :param edges: optional - edges of the graph
        """
        self._connections = connections
        self._n_nodes = connections.shape[0]
        self._node_names = []
        pool = list(ascii_lowercase)
        while len(self._node_names) < self._n_nodes:
            self._node_names = (self._node_names + pool)[:self._n_nodes]
            pool = map(
                lambda x: reduce(
                    op.add,
                    x
                ),
                it.product(pool, ascii_lowercase)
            )

        self.plot(show=True)

        raise NotImplementedError('implement!!!')
        self._fitness = self.fitness

    def __copy__(self):
        raise NotImplementedError('not implemented yet!')

    def __deepcopy__(self, memo):
        raise NotImplementedError('not implemented yet!')

    def randomize_edges(self):
        raise NotImplementedError('not implemented yet!')

    @property
    def fitness(self):
        """
        The fitness of this graph - the number of edges that it's nodes
        doesn't have the same color, normalized to [0,1].

        :rtype: float
        :return: Ranges from 0 (worst fitness) to 1 (best fitness).
        """
        return self._fitness

    @property
    def n_nodes(self):
        """
        :rtype: int
        :return: Number of nodes in this graph.
        """
        return len(self.nodes.keys())

    @property
    def colors(self):
        raise NotImplementedError('not implemented yet!')

    @colors.setter
    def colors(self, value):
        raise NotImplementedError('not implemented yet!')

    def export(self, filename):
        raise NotImplementedError('not implemented yet!')

    @classmethod
    def generate_random(cls, n_nodes):
        triu_x, triu_y = np.triu_indices(n_nodes, 1)  # upper triangular coordinates
        diag_x, diag_y = np.diag_indices(n_nodes)  # diagonal coordinates

        connections = np.empty((n_nodes, n_nodes), dtype=np.int32)
        connections[triu_x, triu_y] = np.random.randint(0, 2, size=len(triu_x), dtype=np.int32)
        connections[triu_y, triu_x] = connections[triu_x, triu_y]
        connections[diag_x, diag_y] = False

        return cls(connections)

    def plot(self, show=False):
        plt.figure()
        G = nx.DiGraph()

        edges = zip(*np.where(self._connections))

        G.add_nodes_from(self._node_names)
        G.add_edges_from(edges)

        layout = nx.circular_layout(G)
        nx.draw_networkx(G, pos=layout, node_color='cyan')
        plt.axis('off')
        # plt.title('GRA')

        if show:
            plt.show()

