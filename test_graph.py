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
from problem import ModelGraph
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex

__author__ = 'Henry Cagnini'


def main():
    n_nodes = 10  # max number of nodes = letters in the alphabet
    n_colors = 2  # number of colors to use in the graph
    n_individuals = 100  # size of the population
    seed = 6  # use None for random or any integer for predetermined randomization
    n_generations = 100  # max iterations to search for optimum

    random.seed(seed)
    np.random.seed(seed)

    available_colors = list(map(
        to_hex,
        cm.viridis(np.linspace(0.5, 1., n_colors))
    ))

    G = ModelGraph.generate_random(n_nodes=n_nodes, available_colors=available_colors)
    G.reset_colors()

    inst = BMDA()

    best, gm = inst.fit(
        modelgraph=G,
        n_individuals=n_individuals,
        n_generations=n_generations
    )

    print('best individual fitness:', best.fitness)
    G.plot(title='Problem')
    best.plot(title='Best individual')
    gm.plot(title='Graphical Model')
    plt.show()


if __name__ == '__main__':
    main()
