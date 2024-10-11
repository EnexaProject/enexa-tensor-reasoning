import numpy as np

from tnreason.engine import subscript_creation as subc


def np_rencoding_from_function(inshape, outshape, incolors, outcolors, function, name="NPEncoding"):
    values = np.zeros(inshape + outshape)
    for i in np.ndindex(*inshape):
        values[i + tuple(
            [int(entry) for entry in function(*i)])] = 1
    return NumpyCore(values=values, colors=incolors + outcolors, name=name)


class NumpyCore:
    def __init__(self, values, colors, name=None):
        self.values = np.array(values)
        self.colors = colors
        self.name = name

        if len(self.colors) != len(self.values.shape):
            raise ValueError("Number of Colors does not match the Value Shape in Core {}!".format(name))
        if len(self.colors) != len(set(self.colors)):
            raise ValueError("There are duplicate colors in the colors {} of Core {}!".format(colors, name))

    def clone(self):
        return NumpyCore(self.values.copy(), self.colors.copy(), self.name)  # ! Shallow Copies?

    ## For Sampling
    def normalize(self):
        return NumpyCore(1 / np.sum(self.values) * self.values, self.colors, self.name)

    ## For ALS: Reorder Colors and summation
    def reorder_colors(self, newColors):
        self.values = np.einsum(subc.get_reorder_substring(self.colors, newColors), self.values)
        self.colors = newColors

    def sum_with(self, sumCore):
        if set(self.colors) != set(sumCore.colors):
            raise ValueError("Colors of summands {} and {} do not match!".format(self.name, sumCore.name))
        else:
            self.reorder_colors(sumCore.colors)
            return NumpyCore(self.values + sumCore.values, self.colors, self.name)

    def multiply(self, weight):
        return NumpyCore(weight * self.values, self.colors, self.name)

    def exponentiate(self):
        return NumpyCore(np.exp(self.values), self.colors, self.name)

    def get_argmax(self):
        return {self.colors[i]: maxPos for i, maxPos in
                enumerate(np.unravel_index(np.argmax(self.values.flatten()), self.values.shape))}

    def draw_sample(self, asEnergy=False, temperature=1):
        if asEnergy:
            distribution = np.exp(self.values * 1 / temperature).flatten()
        else:
            distribution = self.values.flatten()
        sample = np.unravel_index(
            np.random.choice(np.arange(np.prod(distribution.shape)), p=distribution / np.sum(distribution)),
            self.values.shape)
        return {color: sample[i] for i, color in enumerate(self.colors)}

    def calculate_coordinatewise_kl_to(self, secondCore):
        klDivergences = np.empty(self.values.shape)
        for x in np.ndindex(self.values.shape):
            klDivergences[x] = bernoulli_kl_divergence(self.values[x], secondCore.values[x])
        return NumpyCore(values=klDivergences, colors=self.colors, name=str(self.name) + "_kl_" + str(secondCore.name))


class NumpyEinsumContractor:
    def __init__(self, coreDict={}, openColors=[]):
        self.coreDict = {key: coreDict[key].clone() for key in coreDict}
        self.openColors = openColors

    def contract(self):
        substring, coreOrder, colorDict, colorOrder = subc.get_einsum_substring(self.coreDict, self.openColors)
        return NumpyCore(
            np.einsum(substring, *[self.coreDict[key].values for key in coreOrder]),
            [color for color in colorOrder if color in self.openColors])


def bernoulli_kl_divergence(p1, p2):
    """
    Calculates the Kullback Leibler Divergence between two Bernoulli distributions with parameters p1, p2
    """
    if p1 == 0:
        return np.log(1 / (1 - p2))
    elif p1 == 1:
        return np.log(1 / p2)
    return p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))
