from collections import Counter

import numpy as np
import pandas as pd
import itertools as it

__author__ = 'Henry Cagnini'


class Variable(object):
    """
    Each variable can have at most 1 parent.
    """

    def __init__(self, name, values, parent=None):
        """

        :param name:
        :param values:
        :param parent:
        """
        self.name = name
        self.values = values
        self.parent = parent

        vals = it.product(self.values, self.values) if parent is not None else self.values
        val_tuples = []
        for val in vals:
            val_tuples += [(val, )]

        index = pd.MultiIndex.from_tuples(val_tuples)

        self.probs = pd.DataFrame(
            data=1./len(val_tuples),
            index=index,
            columns=['prob'],
        )

        self.__add_rest__()

    def __add_rest__(self):
        summed = np.sum(self.probs)
        rest = 1. - summed
        self.probs.loc[np.random.choice(self.probs.index)] += rest

    def sample(self, evidence=None):
        """

        :param evidence:
        :return:
        """
        if evidence is not None:
            raise NotImplementedError('not implemented yet!')

        a = self.probs.index.values
        p = self.probs.values.ravel()

        color = np.random.choice(a=a, p=p)
        return color

    def is_child_of(self):
        pass

    def is_parent_of(self):
        pass


class GraphicalModel(object):
    def __init__(self, modelgraph):
        """

        :type modelgraph: problem.ModelGraph
        :param modelgraph:
        """

        self.modelgraph = modelgraph

        self.sampling_order = np.arange(self.modelgraph.n_variables)

        self.variables = np.array(
            [Variable(x, self.modelgraph.available_values) for x in self.modelgraph.variable_names],
            dtype=np.object
        )

        self.dependencies = np.zeros((len(self.variables), len(self.variables)), dtype=np.int32)

    def sample(self, n_individuals=1):
        models = []
        for i in xrange(n_individuals):
            values = []
            for var in self.sampling_order:
                val = self.variables[var].sample()
                values += [val]

            models += [self.modelgraph.__class__(
                self.modelgraph.variable_names, self.modelgraph.available_values, self.modelgraph.connections, values
            )]

        if n_individuals == 1:
            return models[0]
        return models

    def update(self, fittest):
        """
        Updates this graphical model, inplace.

        :type fittest: list
        :param fittest:
        :return:
        """
        n_fittest = len(fittest)
        n_variables = fittest[0].n_variables

        genotype = np.empty((n_fittest, n_variables), dtype=np.object)

        for i in xrange(n_fittest):
            genotype[i, :] = fittest[i].colors

        # TODO do!

        self.dependencies[:] = 0

        variable_names = self.modelgraph.variable_names
        N = float(n_fittest)
        expected = map(Counter, genotype.T)

        combs = list(it.product(self.modelgraph.available_values, self.modelgraph.available_values))

        for i in xrange(n_variables):
            for j in xrange(i + 1, n_variables):
                observed = Counter(
                    map(tuple, it.izip(genotype[:, i], genotype[:, j]))
                )

                _sum = 0.
                for a, b in combs:
                    lower = N * (expected[i].get(a, 0.)/N) * (expected[j].get(b, 0.)/N)
                    upper = (N * (((observed.get((a, b), 0.)/N) * (expected[j].get(b, 0.)/N)) - lower)) ** 2.

                    try:
                        _sum += upper / lower
                    except ZeroDivisionError:
                        pass  # sums nothing

                if _sum < 3.84:
                    related = 0
                else:
                    related = 1

                self.dependencies[i, j] = related
                self.dependencies[j, i] = related

        raise NotImplementedError('implementation error!')
        print self.dependencies
        z = 0
