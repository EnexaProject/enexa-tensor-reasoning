from tnreason.contraction import contraction_visualization as cv

from tnreason.engine import subscript_creation as subc

import numpy as np

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']


class TensorCoreBase:
    def __init__(self, values, colors, name=None):
        if len(colors) != len(values.shape):
            raise ValueError("Number of Colors does not match the Value Shape in Core {}!".format(name))
        if len(colors) != len(set(colors)):
            raise ValueError("There are duplicate colors in the colors {} of Core {}!".format(colors, name))

        self.values = values
        self.colors = colors
        self.name = name


class NumpyCore(TensorCoreBase):

    def reduced_contraction(self, core1, reductionColors=[]):
        colorDict = {color: alphabet[i] for i, color in enumerate(np.unique(self.colors + core1.colors))}

        core0String = "".join([colorDict[color] for color in self.colors])
        core1String = "".join([colorDict[color] for color in core1.colors])
        premiseString = ",".join([core0String, core1String])

        outColors = [color for color in np.unique(self.colors + core1.colors) if color not in reductionColors]
        headString = "".join([colorDict[color] for color in outColors])

        contractionString = "->".join([premiseString, headString])

        outValues = np.einsum(contractionString, self.values, core1.values)

        return NumpyCore(outValues, outColors, [self.name, "con", core1.name])

    def reduce_color(self, reductionColor):
        self.reduce_colors([reductionColor])

    def reduce_colors(self, reductionColors):
        colorDict = {color: alphabet[i] for i, color in enumerate(self.colors)}

        contractionString = "->".join([
            "".join([colorDict[color] for color in self.colors]),
            "".join([colorDict[color] for color in self.colors if color not in reductionColors])
        ])

        return NumpyCore(np.einsum(contractionString, self.values),
                         [color for color in self.colors if color not in reductionColors], self.name)

    def get_values_as_array(self):
        return self.values

    def clone(self):
        return NumpyCore(self.values.copy(), self.colors.copy(), self.name)  # ! Shallow Copies?

    ## For Sampling
    def normalize(self):
        return NumpyCore(1 / np.sum(self.values) * self.values, self.colors, self.name)

    ## For ALS
    def reorder_colors(self, newColors):
        oldColors = self.colors.copy()
        oldValues = np.copy(self.values)

        colorDict = {oldColors[i]: alphabet[i] for i in range(len(oldColors))}
        old_string = "".join([colorDict[color] for color in oldColors])
        new_string = "".join([colorDict[color] for color in newColors])
        contraction_string = "->".join([old_string, new_string])

        newValues = np.einsum(contraction_string, oldValues)

        self.values = newValues
        self.colors = newColors

    def sum_with(self, sumCore):
        if set(self.colors) != set(sumCore.colors):
            print(self.colors, sumCore.colors)
            raise ValueError("Colors of summands {} and {} do not match!".format(self.name, sumCore.name))
        else:
            self.reduce_colors(sumCore.colors)
            return NumpyCore(self.values + sumCore.values, self.colors, self.name)

    def multiply(self, weight):
        return NumpyCore(weight * self.values, self.colors, self.name)

    def get_maximal_index(self):
        return np.unravel_index(np.argmax(self.values.flatten()), self.values.shape)


def change_type(cCore, targetType="NumpyTensorCore"):
    if targetType == "NumpyTensorCore":
        return NumpyCore(cCore.values, cCore.colors, cCore.name)
    else:
        raise TypeError("Type {} not understood!".format(targetType))


class NumpyEinsumContractor:
    def __init__(self, coreDict={}, openColors=[], variableNestedList=None, visualize=False,
                 coreType="NumpyTensorCore"):
        if visualize:
            self.visualize(coreDict)

        self.coreDict = {key: coreDict[key].clone() for key in coreDict}
        self.openColors = openColors

        self.variableNestedList = variableNestedList
        self.coreType = coreType

    def create_naive_variableNestedList(self, together=True):
        closedColors = []
        for key in self.coreDict:
            for color in self.coreDict[key].colors:
                if color not in closedColors and color not in self.openColors:
                    closedColors.append(color)
        if together:
            self.variableNestedList = [closedColors]
        else:
            self.variableNestedList = [[color] for color in closedColors]

    def visualize(self, coreDict):
        cv.draw_contractionDiagram(coreDict)

    def contract(self):
        if self.variableNestedList is None:
            self.create_naive_variableNestedList()
        for variables in self.variableNestedList:
            self.contraction_step(variables)
        if len(self.coreDict) == 1:
            return self.coreDict.popitem()[1]
        else:
            return einsum(self.coreDict, [], self.coreType)

    def recursive_contraction(self):
        for variables in self.variableNestedList:
            self.contraction_step(variables)

    def contraction_step(self, variables):
        affectedKeys = [key for key in self.coreDict if not set(self.coreDict[key].colors).isdisjoint(set(variables))]
        contractionCores = {key: self.coreDict.pop(key) for key in affectedKeys}
        self.coreDict["contracted_" + str(variables)] = einsum(contractionCores, variables, self.coreType)


def einsum(conCoreDict, variables, coreType):
    openVariables = []
    for key in conCoreDict:
        for color in conCoreDict[key].colors:
            if color not in variables and color not in openVariables:
                openVariables.append(color)

    substring, coreOrder, colorDict, colorOrder = subc.get_substring(conCoreDict, openVariables)
    return NumpyCore(
        np.einsum(substring, *[conCoreDict[key].values for key in coreOrder]),
        [color for color in colorOrder if color not in variables])


if __name__ == "__main__":
    import tnreason.tensor.formula_tensors as ft

    cores = ft.FormulaTensor(["c", "and", ["not", ["a", "or", "b"]]]).get_cores()
    print(cores)

    contractor = NumpyEinsumContractor(cores, openColors=["a", "b"])
    contractor.create_naive_variableNestedList()
    result = contractor.contract()

    print(result.colors)
    print(result.values)

    print(cores)

    contractor = NumpyEinsumContractor(cores, openColors=["a", "b"])
    contractor.create_naive_variableNestedList()
    result = contractor.contract()

    print(result.colors)
    print(result.values)
