"""
This code exemplifies a simple problem of assigning colors to countries in a world map.

The rule is to NOT allow neighboring countries to have the same color.

The fitness is the number of countries with different colors, normalized by the total number of frontiers.

The user may set any number of colors for this problem. Note that for some cases (such as using only one color)
an effective solution is unfeasible.

This script uses code and information from
https://github.com/FnTm/country-neighbors
"""

import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import json
import pandas as pd
import random
import numpy as np
from BMDA import BMDA
from problem import ModelGraph, WorldMap
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex

__author__ = 'Henry Cagnini'


def plot_worldmap():
    fig = plt.figure(figsize=(600, 400))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1.0)

    # ax.add_feature(cartopy.feature.LAND)
    # ax.add_feature(cartopy.feature.COASTLINE)
    # ax.add_feature(cartopy.feature.LAKES, alpha=0.95)
    # ax.add_feature(cartopy.feature.RIVERS)

    # x_left, x_right, y_down, y_up
    # ax.set_extent([-150, 60, -25, 60])

    shpfilename = shpreader.natural_earth(
        resolution='110m',
        category='cultural',
        name='admin_0_countries'
    )

    reader = shpreader.Reader(shpfilename)
    countries = list(reader.records())

    data = []
    for country in countries:
        data += [[country.attributes['NAME'], country.attributes['ISO_A2'], country.attributes['ISO_A3'], country.attributes['UN_A3']]]
    pd.DataFrame(
        data,
        columns=['name', 'ISO_A2', 'ISO_A3', 'UN_A3']
    ).to_csv('countries.csv', sep=',', quotechar="\"", index=False, encoding='utf-8')

    for country in countries:
        if country.attributes['ADM0_A3'] == 'USA':
            ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                              facecolor=(0, 0, 1),
                              label=country.attributes['ADM0_A3'])
        else:
            ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                              facecolor=(1, 1, 1),
                              label=country.attributes['ADM0_A3']
                             )

    plt.show()


def main():
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

    # TODO fulfill country names and colors!
    G = WorldMap(colors=available_colors)

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
    # plot_worldmap()
    main()

