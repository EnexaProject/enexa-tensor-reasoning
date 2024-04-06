import numpy as np

from tnreason.contraction import contraction_visualization as cv

from tnreason import tensor


class NumpyEinsumContractor:
    def __init__(self, coreDict={}, openColors=[], variableNestedList=None, visualize=False,
                 coreType="NumpyTensorCore"):
        if visualize:
            self.visualize(coreDict)

        self.coreDict = {key : coreDict[key].clone() for key in coreDict}
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
    colorDict = get_colorDict([conCoreDict[key].colors for key in conCoreDict])
    coreOrder = list(conCoreDict.keys())
    colorOrder = list(colorDict.keys())
    leftString = ",".join([
        "".join([colorDict[color] for color in conCoreDict[key].colors])
        for key in coreOrder
    ])
    rightString = "".join([colorDict[color] for color in colorOrder if color not in variables])
    return tensor.get_core(coreType)(
        np.einsum("->".join([leftString, rightString]), *[conCoreDict[key].values for key in coreOrder]),
        [color for color in colorOrder if color not in variables])


def get_colorDict(nestedColorsList):
    symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    colorDict = {}
    i = 0
    for colors in nestedColorsList:
        for color in colors:
            if color not in colorDict:
                if i >= len(symbols):
                    raise ValueError("Length of Contraction is too large for Einsum!")
                colorDict[color] = symbols[i]
                i += 1
    return colorDict


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