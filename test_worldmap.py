"""
This code exemplifies a simple problem of assigning colors to countries in a world map.

The rule is to NOT allow neighboring countries to have the same color.

The fitness is the number of countries with different colors, normalized by the total number of frontiers.

The user may set any number of colors for this problem. Note that for some cases (such as using only one color)
an effective solution is unfeasible.

This script uses code and information from
https://github.com/FnTm/country-neighbors
"""

import random

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from BMDA import BMDA
from problem import WorldMap
import os

__author__ = 'Henry Cagnini'


def main():
    n_colors = 2  # number of colors to use in the graph
    n_individuals = 10  # size of the population
    seed = 6  # use None for random or any integer for predetermined randomization
    n_generations = 2  # max iterations to search for optimum

    random.seed(seed)
    np.random.seed(seed)

    available_colors = list(map(
        to_hex,
        cm.viridis(np.linspace(0.5, 1., n_colors))
    ))

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    connections = pd.read_csv(
        os.path.join(__location__, 'problem', 'country_connections.csv'),
        index_col=0,
        encoding='utf-8',
        keep_default_na=False,
        na_values=[''],
    ).fillna(value=0).astype(np.int32)

    n_countries = len(connections.index)
    G = WorldMap(
        node_names=connections.index,
        available_colors=available_colors,
        connections=connections.values,
        colors=np.repeat(available_colors[0], n_countries).tolist()
    )

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
    # plot_worldmap()
