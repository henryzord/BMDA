# coding=utf-8

import copy

import numpy as np
from datetime import datetime as dt

from GraphicalModel import GraphicalModel

__author__ = 'Henry'


class BMDA(object):
    def __init__(self):
        """
        Initializes the MIMIC Bivariate Marginal Distribution Algorithm.
        """
        pass

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

        t1 = dt.now()
        population = np.array(
            map(
                lambda x: copy.deepcopy(modelgraph),
                xrange(n_individuals)
            )
        )

        gm = GraphicalModel(modelgraph=modelgraph)
        median = np.inf

        for g in xrange(n_generations):
            for i in xrange(n_individuals):
                if population[i].fitness <= median:
                    population[i] = gm.sample()

            fitness = np.array(map(lambda x: x.fitness, population))
            median = np.median(fitness)  # type: float
            fittest_arg = fitness > median

            fittest_mod = population[np.flatnonzero(fittest_arg)]

            t2 = dt.now()
            print 'generation %3.d: (%.6f, %.6f, %.6f) Time elapsed: %.4f' % (
                g, np.min(fitness), median, np.max(fitness), (t2 - t1).total_seconds()
            )
            t1 = dt.now()
            if max(fittest_arg) == 0:  # or converged
                break

            gm.update(fittest_mod)

        fitness = np.array(map(lambda x: x.fitness, population))
        best = np.argmax(fitness)

        return population[best], gm
