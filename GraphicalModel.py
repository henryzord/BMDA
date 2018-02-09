import itertools as it
from collections import Counter

import numpy as np
import pandas as pd
from problem import Graph
import networkx as nx

__author__ = 'Henry Cagnini'


class Variable(object):
    """
    Each variable can have at most 1 parent.
    """
    var_i = 0  # variable index in the probs matrix
    par_i = 1  # parent index in the probs matrix

    def __init__(self, name, values, parent=None):
        """

        :param name:
        :param values:
        :param parent:
        """
        self.name = name
        self.values = values
        self.parent = parent

        vals = it.product(self.values, self.values)
        val_tuples = []
        for val in vals:
            # val_tuples += [(val, )]  # TODO removed
            val_tuples += [val]  # TODO removed

        index = pd.MultiIndex.from_tuples(val_tuples)

        self.probs = pd.DataFrame(
            data=1./len(val_tuples),
            index=index,
            columns=['prob'],
        )

        self.__add_rest__()

    def __add_rest__(self):

        n_index = len(self.probs.index)
        picked = self.probs.index[np.random.choice(n_index)]

        summed = np.sum(self.probs.values.ravel())
        rest = 1. - summed
        if rest > 0:
            self.probs.loc[[picked]] += rest

    def sample(self, evidence=None):
        """

        :param evidence:
        :return:
        """
        if evidence is not None:
            assert self.parent is not None, ValueError('Cannot provide evidence for a variable without parent!')

            a_raw = list(it.product(self.values, [evidence]))
            p_raw = self.probs.loc[a_raw]
            a = p_raw.index.values
            p = p_raw.values.ravel() / p_raw.values.ravel().sum()
            rest = 1. - np.sum(p)
            p[np.random.choice(len(p))] += rest
        else:
            a = self.probs.index.values
            p = self.probs.values.ravel()

        color = np.random.choice(a=a, p=p)[self.var_i]
        return color

    def update(self, observed, parent=None):
        """

        :param parent:
        :type observed: numpy.ndarray
        :param observed: An unidimensional array (in the case of a independent variable) or a bidimensional array
            (where the first column refers to this variable and the second column to the parent) with evidence.
        """
        if isinstance(observed, np.ndarray):
            multidimensional = len(observed.shape) > 1
        elif isinstance(observed, list):
            multidimensional = len(observed[0])
        else:
            raise TypeError('observed must be a list or a numpy.ndarray!')

        if multidimensional:
            observed = list(map(tuple, observed))
            count = Counter(observed)
        else:
            combs = it.product(self.values, self.values)
            count = dict()
            raw_count = Counter(observed)

            for var, par in combs:
                count[(var, par)] = float(raw_count.get(var, 0.))

        for (var, par), v in count.items():
            self.probs.loc[[(var, par)]] = v

        self.parent = parent

        self.probs /= np.sum(self.probs)

        self.__add_rest__()


class GraphicalModel(Graph):
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

    def sample(self, individual=None):
        values = dict()
        for var in self.sampling_order:
            variable = self.variables[var]
            parent = variable.parent
            evidence = values[parent] if parent is not None else None

            val = variable.sample(evidence)
            values[variable.name] = val

        ordered_values = sorted(values.items(), key=lambda x: x[0])
        names, sampled = zip(*ordered_values)

        individual.colors = list(sampled)

        return individual

    @staticmethod
    def __check_correlation__(available_values, genotype, correlation):
        """
        Checks whether the variables of the problem are correlated. Adapted from
        Martin Pelikan, Heinz Muehlenbein. The Bivariate Marginal Distribution Algorithm.

        :type available_values: list
        :param available_values: Available values that each variable can assume.
            Currently, all variables must assume the same values.
        :type genotype: numpy.ndarray
        :param genotype: Population to observe, where each row is an individual and each column a variable.
        :type correlation: numpy.ndarray
        :param correlation: A float matrix with dimensions len(available_values) x len(available_values),
            where the values denote the value in the chi-squared distribution (>= 3.84 means correlation for at least
            95% of the occurrences).
        :rtype: numpy.ndarray
        :return: The correlation matrix updated.
        """

        n_fittest, n_variables = genotype.shape

        correlation[:] = 0

        N = float(n_fittest)
        expected = list(map(Counter, genotype.T))

        combs = list(it.product(available_values, available_values))

        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                observed = Counter(
                    map(tuple, zip(genotype[:, i], genotype[:, j]))
                )

                chi = 0.
                for a, b in combs:
                    p_ia = expected[i].get(a, 0.) / N
                    p_jb = expected[j].get(b, 0.) / N

                    try:
                        p_ia_jb = (observed.get((a, b), 0.) / float(expected[j].get(b, 0.))) * p_jb
                    except ZeroDivisionError:
                        p_ia_jb = 0.

                    lower = N * (p_ia * p_jb)
                    upper = (N * p_ia_jb - lower) ** 2

                    try:
                        chi += upper / lower
                    except ZeroDivisionError:
                        pass  # sums nothing

                correlation[i, j] = chi
                correlation[j, i] = chi

        return correlation

    def update(self, fittest):
        """
        Updates this graphical model, inplace.

        :type fittest: list
        :param fittest:
        :return:
        """

        n_fittest = len(fittest)
        genotype = np.empty((n_fittest, self.modelgraph.n_variables), dtype=np.object)
        for i in range(n_fittest):
            genotype[i, :] = fittest[i].colors

        self.dependencies = self.__check_correlation__(self.modelgraph.available_values, genotype, self.dependencies)

        n_variables = self.modelgraph.n_variables

        self.sampling_order[:] = -1

        arg_i = np.random.choice(n_variables)
        self.sampling_order[0] = arg_i
        self.variables[arg_i].update(genotype[:, arg_i])
        n_added = 1

        G = {arg_i}
        A = {i for i in range(n_variables)} - G  # not yet in the dependence graph

        while n_added < n_variables:
            max_chi = -np.inf
            arg_max = -1
            arg_i = -1
            for i in G:
                arg_chi = list(A)[np.argmax(self.dependencies[i, list(A)])]
                if self.dependencies[i, arg_chi] > max_chi:
                    max_chi = self.dependencies[arg_chi, i]
                    arg_max = arg_chi
                    arg_i = i

            if (max_chi >= 3.84) and (arg_i != arg_max):
                self.variables[arg_max].update(observed=genotype[:, [arg_max, arg_i]], parent=self.variables[arg_i].name)
            else:
                arg_max = np.random.choice(list(A))
                self.variables[arg_max].update(observed=genotype[:, arg_max], parent=None)
            G |= {arg_max}
            A -= {arg_max}

            self.sampling_order[n_added] = arg_max
            n_added += 1

    def plot(self, title=''):
        super(GraphicalModel, self).plot(title)

        G = nx.DiGraph()

        parents = dict()
        sole = []
        for variable in self.variables:
            if variable.parent is None:
                sole += [variable.name]
            else:
                parents[variable.name] = variable.parent

        G.add_nodes_from(sole)
        G.add_edges_from(
            map(lambda x: x[::-1], parents.items())
        )
        # layout = nx.circular_layout(G)
        nx.draw_networkx(G, node_color='cyan')  # pos=layout)
