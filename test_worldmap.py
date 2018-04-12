"""
This code exemplifies a simple problem of assigning colors to countries in a world map.

The rule is to NOT allow neighboring countries to have the same color.

The fitness is the number of countries with different colors, normalized by the total number of frontiers.

The user may set any number of colors for this problem. Note that for some cases (such as using only one color)
an effective solution is unfeasible.

This script uses code and information from
https://github.com/FnTm/country-neighbors
"""

import os
import pickle
import random

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from plotly.graph_objs import *
from plotly.offline import plot

from BMDA import BMDA
from problem import WorldMap

__author__ = 'Henry Cagnini'


def main(n_colors, n_individuals, n_generations, random_state=None):
    random.seed(random_state)
    np.random.seed(random_state)

    available_colors = list(map(
        to_hex,
        cm.viridis(np.linspace(0., 1., n_colors))
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

    # pickle.dump(best, open('best.bin', 'wb'))
    # pickle.dump(gm, open('gm.bin', 'wb'))

    print('best individual fitness:', best.fitness)
    # G.plot(title='Problem')
    best.plot(title='Best individual')
    # gm.plot(title='Graphical Model')
    plt.show()


def plotly_plotting(best, gm):
    """

    :param gm:
    :type gm: GraphicalModel.GraphicalModel
    :type best: WorldMap
    :param best:
    :return:
    """

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # gt stands for ground truth connections, i.e. the real connections of a country to its neighbors
    gt_connections = pd.read_csv(
        os.path.join(__location__, 'problem', 'country_connections.csv'),
        index_col=0,
        encoding='utf-8',
        keep_default_na=False,
        na_values=[''],
    ).fillna(value=0).astype(np.int32)  # type: pd.DataFrame

    country_codes = pd.read_csv(
        os.path.join(__location__, 'problem', 'country_codes.csv'),
        encoding='utf-8',
        keep_default_na=False,
        na_values=[''],
    )

    geometries = pickle.load(open(os.path.join(__location__, 'problem', 'country_geometries.bin'), 'rb'))

    isoa2_to_name = dict(list(zip(country_codes['ISO_A2'], country_codes['NAME'])))

    node_names = best.node_names

    gt_nonzero = []
    for index, content in gt_connections.iterrows():
        for column in gt_connections.columns:
            if gt_connections.at[index, column] != 0:
                gt_nonzero += [(index, column)]

    gt_G = nx.Graph(gt_nonzero)  # type: nx.Graph
    gt_G.add_nodes_from(node_names)

    # best stands for the best individual found during the evolutionary process
    best_G = nx.DiGraph()

    parents = dict()
    for variable in gm.variables:
        if variable.parent is not None:
            parents[variable.name] = variable.parent

    best_G.add_nodes_from(node_names)
    best_G.add_edges_from(
        map(lambda x: x[::-1], parents.items())
    )

    for node_name in node_names:
        gt_G.nodes[node_name]['pos'] = (geometries[node_name].geometry.centroid.x, geometries[node_name].geometry.centroid.y)
        best_G.nodes[node_name]['pos'] = (geometries[node_name].geometry.centroid.x, geometries[node_name].geometry.centroid.y)

    gt_edges = dict(x=[], y=[])
    for edge in gt_G.edges():
        x0, y0 = gt_G.node[edge[0]]['pos']
        x1, y1 = gt_G.node[edge[1]]['pos']
        gt_edges['x'] += [x0, x1, None]
        gt_edges['y'] += [y0, y1, None]

    gt_edges = Scatter(
        name='ground truth',
        x=gt_edges['x'],
        y=gt_edges['y'],
        line=Line(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        visible='legendonly'
    )

    best_edges = dict(x=[], y=[])
    neighbors = {x: [] for x in node_names}
    for edge in best_G.edges():
        x0, y0 = best_G.node[edge[0]]['pos']
        x1, y1 = best_G.node[edge[1]]['pos']
        best_edges['x'] += [x0, x1, None]
        best_edges['y'] += [y0, y1, None]

        neighbors[edge[0]] += [isoa2_to_name[edge[1]]]
        neighbors[edge[1]] += [isoa2_to_name[edge[0]]]

    best_edges = Scatter(
        name='best solution',
        x=best_edges['x'],
        y=best_edges['y'],
        line=Line(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    country_x = []
    country_y = []
    country_name = []
    country_colors = best.colors
    for node in best_G.nodes():
        country_x += [best_G.node[node]['pos'][0]]
        country_y += [best_G.node[node]['pos'][1]]

        country_name += [isoa2_to_name[node] + '<br>connected to:<br>' + '<br>'.join(neighbors[node])]

    node_trace = Scatter(
        name='countries',
        x=country_x,
        y=country_y,
        text=country_name,
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            showscale=False,
            colorscale='Viridis',
            reversescale=True,
            color=country_colors,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        )
    )

    layout = Layout(
        title='<br>Network graph made with Python',
        titlefont=dict(size=16),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> "
                     "https://plot.ly/ipython-notebooks/network-graphs/</a>",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            )
        ],
        xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig = Figure(
        data=[gt_edges, best_edges, node_trace],
        layout=layout
    )

    plot(fig, filename='networkx.html')


if __name__ == '__main__':
    _n_colors = 16  # number of colors to use in the graph
    _n_individuals = 50  # size of the population
    _random_state = 6  # use None for random or any integer for predetermined randomization
    _n_generations = 10  # max iterations to search for optimum

    main(n_colors=_n_colors, n_individuals=_n_individuals, n_generations=_n_generations, random_state=_random_state)
    # _gm = pickle.load(open('gm.bin', 'rb'))
    # _best = pickle.load(open('best.bin', 'rb'))
    # plotly_plotting(_best, _gm)
