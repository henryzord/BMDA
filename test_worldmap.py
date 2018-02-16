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

from plotly.offline import plot
import plotly.plotly as py
from plotly.graph_objs import *
import networkx as nx
import pickle

__author__ = 'Henry Cagnini'


def main():
    n_colors = 16  # number of colors to use in the graph
    n_individuals = 50  # size of the population
    seed = 6  # use None for random or any integer for predetermined randomization
    n_generations = 10  # max iterations to search for optimum

    random.seed(seed)
    np.random.seed(seed)

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

    print('best individual fitness:', best.fitness)
    G.plot(title='Problem')
    best.plot(title='Best individual')
    gm.plot(title='Graphical Model')
    plt.show()


def plotly_plotting():
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    connections = pd.read_csv(
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

    nonzero = []
    for index, content in connections.iterrows():
        for column in connections.columns:
            if connections.at[index, column] != 0:
                nonzero += [(index, column)]

    G = nx.Graph(nonzero)  # type: nx.Graph

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    countries = pickle.load(open(os.path.join(__location__, 'problem', 'country_geometries.bin'), 'rb'))

    isoa2_to_name = dict(list(zip(country_codes['ISO_A2'], country_codes['NAME'])))

    for node_name in G.nodes:
        G.nodes[node_name]['pos'] = (countries[node_name].geometry.centroid.x, countries[node_name].geometry.centroid.y)

    # pos = nx.get_node_attributes(G, 'pos')

    edge_trace = Scatter(
        x=[],
        y=[],
        line=Line(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=[],
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

    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)

    for node, adjacencies in G.adjacency():
        node_trace['marker']['color'].append(len(adjacencies))
        node_info = isoa2_to_name[node] + '<br>' + '# of connections: ' + str(len(adjacencies))
        node_trace['text'].append(node_info)

    layout = Layout(
        title='<br>Network graph made with Python',
        titlefont=dict(size=16),
        # showlegend=False,
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

    # traces = Data([edge_trace, node_trace], name='ground truth')
    traces = Data([edge_trace, node_trace], name='ground truth')

    fig = Figure(
        data=traces,
        layout=layout
    )

    plot(fig, filename='networkx.html')


if __name__ == '__main__':
    # main()
    plotly_plotting()
