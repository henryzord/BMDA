"""
This code exemplifies a simple problem of assigning colors to nodes in a graph.

The rule is to NOT allow neighboring nodes to have the same color.

The fitness is the number of edges which connect nodes with different colors, normalized by the total number of edges.

The user may set any number of nodes or colors for the problem. Note that for some cases (such as using only one color)
an effective solution is unfeasible.

"""

import random
import numpy as np

from BMDA import BMDA
from ModelGraph import ModelGraph

__author__ = 'Henry Cagnini'


def main():
    n_nodes = 5  # max number of nodes = letters in the alphabet
    n_colors = 3  # number of colors to use in the graph
    n_individuals = 1000  # size of the population
    seed = None  # use None for random or any integer for predetermined randomization
    max_iter = 1000  # max iterations to search for optimum

    random.seed(seed)
    np.random.seed(seed)

    G = ModelGraph.generate_random(n_nodes=n_nodes)

    # plot_random_graph(G)

    inst = BMDA(n_individuals, G, n_colors, max_iter)
    inst.solve()
    inst.summary(screen=True, pdf=False, file=False)


if __name__ == '__main__':
    main()
