import numpy as np

from tnreason.engine import subscript_creation as subc


class NumpyCore:
    def __init__(self, values, colors, name=None):
        self.values = np.array(values)
        self.colors = colors
        self.name = name

        if len(self.colors) != len(self.values.shape):
            raise ValueError("Number of Colors does not match the Value Shape in Core {}!".format(name))
        if len(self.colors) != len(set(self.colors)):
            raise ValueError("There are duplicate colors in the colors {} of Core {}!".format(colors, name))

    def get_values_as_array(self):
        return self.values

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
            print(self.colors, sumCore.colors)
            raise ValueError("Colors of summands {} and {} do not match!".format(self.name, sumCore.name))
        else:
            self.reorder_colors(sumCore.colors)
            return NumpyCore(self.values + sumCore.values, self.colors, self.name)

    def multiply(self, weight):
        return NumpyCore(weight * self.values, self.colors, self.name)

    def get_maximal_index(self):
        return np.unravel_index(np.argmax(self.values.flatten()), self.values.shape)

    def draw_sample(self, asEnergy=False, temperature=1):
        if asEnergy:
            distribution = np.exp(self.values * 1/temperature).flatten()
        else:
            distribution = self.values.flatten()
        sample = np.unravel_index(
            np.random.choice(np.arange(np.prod(distribution.shape)), p=distribution / np.sum(distribution)),
            self.values.shape)
        return {color: sample[i] for i, color in enumerate(self.colors)}


class NumpyEinsumContractor:
    def __init__(self, coreDict={}, openColors=[]):
        self.coreDict = {key: coreDict[key].clone() for key in coreDict}
        self.openColors = openColors

    def contract(self):
        substring, coreOrder, colorDict, colorOrder = subc.get_einsum_substring(self.coreDict, self.openColors)
        return NumpyCore(
            np.einsum(substring, *[self.coreDict[key].values for key in coreOrder]),
            [color for color in colorOrder if color in self.openColors])
