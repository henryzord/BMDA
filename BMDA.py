# coding=utf-8

import sys
import copy
import graphviz
import operator
import itertools
import numpy as np
import networkx as nx
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
import pandas as pd
import itertools as it

from GraphicalModel import GraphicalModel

__author__ = 'Henry'


class BMDA(object):
    def __init__(self):
        """
        Initializes the MIMIC Bivariate Marginal Distribution Algorithm.
        """

    def fit(self, modelgraph, n_individuals=100, n_generations=100):
        """
        Solves the k-max coloring problem.

        :type modelgraph: Problem.Problem
        :param modelgraph: problem to be optimized. Must be descendent of a the problem class for having the appropriate
            interface.

        :type n_individuals: int
        :param n_individuals: optional - The size of the population. Defaults to 100.

        :type n_generations: int
        :param n_generations: optional - Max number of iterations. Defaults to 100.
        """

        population = np.array(
            map(
                lambda x: copy.deepcopy(modelgraph),
                xrange(n_individuals)
            )
        )

        resample = np.ones(n_individuals, dtype=np.bool)
        gm = GraphicalModel(modelgraph=modelgraph)

        g = 0
        while g < n_generations:
            for i in xrange(n_individuals):
                if resample[i]:
                    population[i] = gm.sample()

            fitness = np.array(map(lambda x: x.fitness, population))
            median = np.median(fitness)

            fittest_arg = fitness > median

            if max(fittest_arg) == 0:
                break

            fittest_mod = population[np.flatnonzero(fittest_arg)]

            gm.update(fittest_mod)

            # if gm.converged():
            #     break

            g += 1

        fitness = np.array(map(lambda x: x.fitness, population))
        best = np.argmax(fitness)

        return population[best]

    # def summary(self, screen=True, file=False, pdf=False):
    #     _str = 'Finished inference in ' + str(self.last_iteration) + ' iterations.\n'
    #     _str += 'Evaluations: ' + str(self.last_iteration * self.n_individuals) + '\n'
    #     _str += 'Population size: ' + str(self.n_individuals) + '\n'
    #     _str += 'Nodes: ' + str(self.model_graph.n_nodes) + '\n'
    #     _str += 'Colors: ' + str(len(self.palette)) + '\n'
    #     _str += 'Best individual fitness: ' + str(round(self.__best_individual__().__get_fitness__, 2)) + "\n"
    #
    #     print _str
    #
    #     _dict = copy.deepcopy(self.dependencies)
    #     del _dict[None]
    #
    #     if pdf:
    #         bbn = graphviz.Digraph()
    #         [bbn.node(x) for x in self.node_names]
    #
    #         for parent, children in _dict.iteritems():
    #             for child in children:
    #                 bbn.edge(parent, child)
    #
    #         bbn.render('bbn.gv')
    #
    #         self.__best_individual__().export('optimal')
    #
    #     if file:
    #         with open('output.txt', 'w') as wfile:
    #             wfile.write(_str)
    #
    #     if screen:
    #         def plot_bayseian_network():
    #             plt.figure(1)
    #             G = nx.DiGraph()
    #
    #             G.add_nodes_from(self.node_names)
    #
    #             for parent, children in _dict.iteritems():
    #                 for child in children:
    #                     G.add_edge(parent, child)
    #
    #             layout = nx.circular_layout(G)
    #             nx.draw_networkx(G, pos=layout, cmap=plt.get_cmap('jet'), node_color=list(itertools.repeat('cyan', len(self.node_names))))
    #             plt.axis('off')
    #             plt.title('Bayesian Network')
    #
    #         def plot_best_individual():
    #             plt.figure(2)
    #             individual = self.__best_individual__()
    #             G = nx.Graph()
    #             G.add_nodes_from(individual.names)
    #             some_edges = [tuple(list(x)) for x in individual.edges]
    #             G.add_edges_from(some_edges)
    #             layout = nx.fruchterman_reingold_layout(G)
    #             nx.draw_networkx(G, pos=layout, cmap=plt.get_cmap('jet'), node_color=individual.colors.values())
    #             plt.axis('off')
    #             plt.title('Best Individual')
    #
    #         plot_bayseian_network()
    #         plot_best_individual()
    #         plt.show()
