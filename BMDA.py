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

from ModelGraph import ModelGraph

__author__ = 'Henry'


def declare_function(header, body):
    """
    Declares the header function using the body code.

    :type header: str
    :param header: The name of the function.
    :type body: str
    :param body: The code of the function.
    :rtype: function
    :return: A pointer to the function.
    """
    exec body
    return locals()['f_' + str(header)]


class BMDA(object):
    population_size = 0
    model_graph = None
    palette = []
    population = []
    bbn = None

    dependencies = []
    _bbn_functions = []
    node_names = []

    last_iteration = None

    def __init__(self, population_size, model_graph, n_colors, max_iter=1000):
        """
        Initializes the MIMIC Bivariate Marginal Distribution Algorithm.

        :type population_size: int
        :param population_size: The size of the population.

        :type model_graph: ModelGraph
        :param model_graph: The ModelGraph which the population will be copied of.

        :type n_colors: int
        :param n_colors: Number of colors to use in the graph.

        :param max_iter: optional - Max number of iterations. Defaults to 1000.
        """

        self.population_size = population_size
        self.model_graph = model_graph
        self.palette = cm.viridis(np.linspace(0.5, 1., n_colors))

        self.node_names = self.model_graph.names
        self.max_iter = max_iter

        self.population = np.array(
            map(
                lambda x: copy.deepcopy(self.model_graph),
                xrange(self.population_size)
            )
        )

        self.dependencies = {None: self.node_names}
        self.__build_bbn__(depends_on=None)

    def solve(self):
        """
        Solves the k-max coloring problem.
        """

        i = 1
        least_fit = []
        while i <= self.max_iter:
            self.__sample__(least_fit)
            fitness = map(lambda x: x.fitness, self.population)
            median = np.median(fitness)
            fittest = list(itertools.ifilter(lambda x: x.fitness >= median, self.population))
            least_fit = list(itertools.ifilter(lambda x: x.fitness < median, self.population))

            # former depends on latter in the tuple
            self.dependencies = self.__search_dependencies__(fittest)
            self.__build_bbn__(depends_on=self.dependencies, fittest=fittest)

            if self.__has_converged__():
                break

            sys.stdout.write('\r' + 'Iterations: ' + "%03d" % (i,))
            self.last_iteration = i
            i += 1

        print '\n'

    def __build_bbn__(self, depends_on=None, fittest=[]):
        """
        Build a bayesian belief network and sets it to self._bbn_functions.

        :type depends_on: dict
        :param depends_on: optional - the dependency chain between attributes.

        :type fittest: list
        :param fittest: A list of the fittest individuals (denoted as ModelGraph's) for this generation.
        """
        functions = []
        if not depends_on:
            # will build a simple chain dependency
            for node in self.node_names:
                _str = "def f_" + node + "(" + node + "):\n    return " + str(1. / float(len(self.palette)))
                func = declare_function(node, _str)
                functions += [(node, func)]

        else:
            # delete former functions to avoid any overlap in the next executions
            if len(self._bbn_functions) > 0:
                for func in self._bbn_functions:
                    del func

            # fittest_dict is a collection of the fittest individuals, grouped by their attributes in a dictionary
            fittest_dict = BMDA.__rotate_dict__(map(lambda x: x.colors, fittest), dict_to_list=False)
            functions = []
            count_fittest = len(fittest)

            for attribute in self.node_names:
                parent = list(itertools.ifilter(lambda x: attribute in x[1], self.dependencies.iteritems()))[0][0]
                _str = self.__build_function__(attribute, count_fittest, fittest_dict, dependency=parent)
                func = declare_function(attribute, _str)
                functions += [(attribute, func)]

        self._bbn_functions = dict(functions)

    def __build_function__(self, drawn, count_fittest, fittest_dict, dependency=None):
        """
        Builds a function, which later will be used to infer values for attributes.

        :type drawn: str
        :param drawn: The attribute which function will be defined.

        :type count_fittest: int
        :param count_fittest: The number of fittest individuals for this generation.

        :type fittest_dict: dict
        :param fittest_dict: The fittest individuals grouped by attributes.

        :type dependency: str
        :param dependency: the attribute which the drawn attribute depends on, or None otherwise.

        :rtype: str
        :return: The function body.
        """

        carriage = "    "
        _str = "def f_" + drawn + "(" + drawn + (', ' + dependency if dependency else '') + "):\n"

        if not dependency:
            count_drawn = Counter(fittest_dict[drawn])

            for i, color in enumerate(self.palette):
                _str += carriage + "if " + drawn + " == '" + color + "':\n" + (carriage * 2) + "return " + \
                    str(float(count_drawn.get(color) if color in count_drawn else 0.) / float(count_fittest)) + "\n"

        else:
            iterated = itertools.product(self.palette, self.palette)
            count_dependency = Counter(zip(fittest_dict[drawn], fittest_dict[dependency]))

            for ct in iterated:
                denominator = np.sum(map(lambda x: count_dependency.get(x) if x in count_dependency else 0., itertools.product(self.palette, [ct[1]])))

                value = 0. if denominator == 0. else float(count_dependency.get((ct[0], ct[1])) if (ct[0], ct[1]) in count_dependency else 0.) / denominator

                _str += carriage + "if " + drawn + " == '" + ct[0] + "' and " + dependency + " == '" + ct[1] + "':\n"
                _str += (carriage * 2) + "return " + str(value) + "\n"

        return _str

    def __sample__(self, least_fit=[]):
        """
        Assigns colors to the population of graphs.

        :type least_fit: list
        :param least_fit: optional - the sample to be overrided. If not provided, will replace the whole population.
        """
        sample = dict()
        children = list(itertools.product(self.dependencies[None], [None]))

        if not least_fit:
            size = self.population_size
            population = self.population
        else:
            size = len(least_fit)
            population = least_fit

        while len(children) > 0:
            # current[1] is the parent; current[0] is the child
            current = children[0]  # first child in the list

            probabilities = []
            if current[1] not in sample:
                for color in self.palette:
                    probabilities.append(self._bbn_functions[current[0]](color))
            else:
                # _product = itertools.product(self.palette, sample[current[1]])
                raise NameError('implement me!')

            sample[current[0]] = np.random.choice(self.palette, size=size, replace=True, p=probabilities)
            children.remove(current)

        # rotates the dictionary
        sample = BMDA.__rotate_dict__(sample, dict_to_list=True)

        for graph, colors in itertools.izip(population, sample):
            graph.colors = colors

    @staticmethod
    def __rotate_dict__(sample, dict_to_list):
        if dict_to_list:
            return map(
                lambda x: dict(itertools.izip(sample.iterkeys(), x)),
                itertools.izip(*sample.values())
            )
        else:
            keys = sample[0].keys()
            # initializes an empty sample_dict
            sample_dict = dict(map(lambda x: (x, []), keys))
            for individual in sample:  # iterates over individuals in the sample
                for key, color in individual.iteritems():  # iterates over colors in the individual
                    sample_dict[key].append(color)

            return sample_dict

    def __search_dependencies__(self, fittest):
        """
        Infers the dependencies between attributes.

        :type fittest: list
        :param fittest: A list of the fittest individuals for this generation.

        :rtype: dict
        :return: A list of tuples containing each pair of dependencies.
        """

        fittest_dict = self.__rotate_dict__(map(lambda x: x.colors, fittest), dict_to_list=False)

        # all possible edges, V x V with repeated elements
        # since the dependency may be in either way
        all_edges = filter(lambda (i, j): i != j, itertools.product(self.node_names, self.node_names))
        D = dict()  # set of dependent vertices
        D_strength = dict()

        for (i, j) in all_edges:
            d = self.__chi_squared__(fittest_dict[i], fittest_dict[j], len(fittest))

            # if the dependence holds for 95% of the population
            # and it was not yet defined
            # and does not form a cycle
            if d >= 3.84:  # and (True if (i not in (D[j] if j in D else [])) else False):
                if i in D:
                    D[i] += [j]
                else:
                    D[i] = [j]
                D_strength[(i, j)] = d

        A = set(self.node_names)
        R = set()
        E = {None: []}

        while any(A):
            a = np.random.choice(list(A))
            A -= {a}

            if not any(A):
                E[None] += [a]
                break

            links = filter(lambda f: f in R and a in D[f], D)
            if any(links):
                links_strength = dict(map(lambda f: (D_strength[(f, a)], f), links))
                strongest = links_strength[max(links_strength)]
                if strongest in E:
                    E[strongest] += [a]
                else:
                    E[strongest] = [a]
            else:
                E[None] += [a]

            R |= {a}

        return E

    @staticmethod
    def __remove_cycles__(E):
        """
        Removes cycles from a Bayesian Network.

        :type E: dict
        :param E: The bayesian network denoted as a dictionary.

        :rtype: dict
        :return: The same bayesian network, without cycles.
        """

        for node in E:
            for child in copy.deepcopy(E[node]):
                if BMDA.__has_cycle__(E, child, visited=[node]):
                    E[node].remove(child)

                    if child not in reduce(operator.add, E.values()):
                        E[None] += child

        E = dict(filter(lambda (y, x): any(x), E.items()))
        return E

    @staticmethod
    def __has_cycle__(E, current, visited=[]):
        if current in visited:
            return True
        if current not in E or not any(E[current]):
            return False

        return reduce(
                operator.and_,
                map(
                    lambda x: BMDA.__has_cycle__(E, x, visited + [current]),
                    E[current]
                )
            )

    @staticmethod
    def __marginalize__(dependencies, p_value):
        """
        Marginalizes the distribution given the p_value: P(p_value | any_q_value). Returns the frequency.

        :type dependencies: Counter
        :param dependencies: An instance of the class collections.Counter, with each configuration of values and
            its values. The tuples must be in the for of (p, q).

        :type p_value: Any
        :param p_value: The value of the conditioned variable.

        :rtype: float
        :return: The probability that p assumes the provided value given q values.
        """
        keys = []
        for item in dependencies.keys():
            if item[0] == p_value:
                keys.append(item)

        total = float(reduce(operator.add, dependencies.values()))
        _sum = np.sum(map(lambda x: dependencies[x], keys)) / total
        return _sum

    def __chi_squared__(self, X_i, X_j, N):
        """
        Calculates the Chi squared statistics (the probability that two attributes are independent)
            over two attributes.

        :type X_i: list
        :param X_i: The list of i-th attributes of the fittest individuals.

        :type X_j: list
        :param X_j: The list of j-th attributes of the fittest individuals.

        :rtype: float
        :return: The chi squared statistics.
        """

        joint_sets = Counter(itertools.izip(X_i, X_j))
        i_sets = Counter(X_i)
        j_sets = Counter(X_j)

        index = 0.
        for (y_i, y_j) in itertools.product(self.palette, self.palette):
            p_y_i_y_j = float(joint_sets.get((y_i, y_j))) / N if (y_i, y_j) in joint_sets else 0.
            p_y_i = float(i_sets.get(y_i)) / N if y_i in i_sets else 0.
            p_y_j = float(j_sets.get(y_j)) / N if y_j in j_sets else 0.

            _index = float(((N * p_y_i_y_j) - (N * p_y_i * p_y_j)) ** 2.) / float(N * p_y_i * p_y_j) if N * p_y_i * p_y_j > 0. else 0.
            index += _index

        return index

    def __entropy__(self, attribute, sample, free=None):
        """
        Calculates the entropy of a given attribute.
        :type attribute: str
        :param attribute: The attribute name.

        :type free: str
        :param free: optional -- If the attribute is dependent on other attribute. In this case,
            it shall be provided the name of the free attribute.

        :rtype: tuple
        :return: A tuple containing the name of the attribute alongside its entropy.
        """
        if not free:
            return attribute, -1. * np.sum(
                map(
                    lambda x: (float(x) / len(sample)) * np.log2(float(x) / len(sample)),
                    Counter(map(lambda y: y.nodes[attribute].color, sample)).values()
                )
            )
        else:
            conditionals = Counter(map(lambda x: (x.nodes[attribute].color, x.nodes[free].color), sample))

            entropy = 0.
            for value in set(
                    map(lambda x: x[0], conditionals.keys())):  # iterates over the values of the conditioned attribute
                marginal = self.__marginalize__(conditionals, value)
                entropy += marginal * np.log2(marginal)

            return (attribute, free), -1. * entropy

    def __has_converged__(self):
        fitness = map(
            lambda x: x.fitness,
            self.population
        )
        median = np.median(fitness)
        result = np.all(fitness == median)
        return result

    def __best_individual__(self):
        fitness = map(lambda x: x.fitness, self.population)
        return self.population[np.argmax(fitness)]

    def summary(self, screen=True, file=False, pdf=False):
        _str = 'Finished inference in ' + str(self.last_iteration) + ' iterations.\n'
        _str += 'Evaluations: ' + str(self.last_iteration * self.population_size) + '\n'
        _str += 'Population size: ' + str(self.population_size) + '\n'
        _str += 'Nodes: ' + str(self.model_graph.n_nodes) + '\n'
        _str += 'Colors: ' + str(len(self.palette)) + '\n'
        _str += 'Best individual fitness: ' + str(round(self.__best_individual__().fitness, 2)) + "\n"

        print _str

        _dict = copy.deepcopy(self.dependencies)
        del _dict[None]

        if pdf:
            bbn = graphviz.Digraph()
            [bbn.node(x) for x in self.node_names]

            for parent, children in _dict.iteritems():
                for child in children:
                    bbn.edge(parent, child)

            bbn.render('bbn.gv')

            self.__best_individual__().export('optimal')

        if file:
            with open('output.txt', 'w') as wfile:
                wfile.write(_str)

        if screen:
            def plot_bayseian_network():
                plt.figure(1)
                G = nx.DiGraph()

                G.add_nodes_from(self.node_names)

                for parent, children in _dict.iteritems():
                    for child in children:
                        G.add_edge(parent, child)

                layout = nx.circular_layout(G)
                nx.draw_networkx(G, pos=layout, cmap=plt.get_cmap('jet'), node_color=list(itertools.repeat('cyan', len(self.node_names))))
                plt.axis('off')
                plt.title('Bayesian Network')

            def plot_best_individual():
                plt.figure(2)
                individual = self.__best_individual__()
                G = nx.Graph()
                G.add_nodes_from(individual.names)
                some_edges = [tuple(list(x)) for x in individual.edges]
                G.add_edges_from(some_edges)
                layout = nx.fruchterman_reingold_layout(G)
                nx.draw_networkx(G, pos=layout, cmap=plt.get_cmap('jet'), node_color=individual.colors.values())
                plt.axis('off')
                plt.title('Best Individual')

            plot_bayseian_network()
            plot_best_individual()
            plt.show()
