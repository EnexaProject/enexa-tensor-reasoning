import numpy as np
from tnreason.contraction import generic_cores as gc


class VariableEliminator:
    def __init__(self, coreDict={}, variablesList=[], openColors=[]):
        self.coreDict = coreDict
        self.variablesList = variablesList
        self.openColors = openColors

    def get_affected_cores(self, variables):
        affectedCores = []
        for key in self.coreDict:
            for color in self.coreDict[key].colors:
                if color in variables:
                    affectedCores.append(key)
                    break
        return affectedCores

    def optimize_variableList(self):
        pass

    def contract(self):
        for variables in self.variablesList:
            affectedCores = self.get_affected_cores(variables)
            contracted = variables_contraction({key: self.coreDict.pop(key) for key in affectedCores}, variables)
            self.coreDict["contracted"] = contracted


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']


def get_colorDict(coreDict):
    colorDict = {}
    alphabetCounter = 0
    for key in coreDict:
        for color in coreDict[key].colors:
            if color not in colorDict:
                colorDict[color] = alphabet[alphabetCounter]
                alphabetCounter += 1
    return colorDict


def get_out_colors(coreDict, variables):
    outColors = []
    for key in coreDict:
        for color in coreDict[key].colors:
            if color not in outColors and color not in variables:
                outColors.append(color)
    return outColors


def get_colorString(coreColors, colorDict):
    return "".join(colorDict[color] for color in coreColors)


def variables_contraction(coreDict, contractionVariables, name="Contracted"):
    colorDict = get_colorDict(coreDict)
    outColors = get_out_colors(coreDict, contractionVariables)

    inString = ",".join(get_colorString(coreDict[key].colors, colorDict) for key in coreDict)
    outString = get_colorString(outColors, colorDict)
    conString = "->".join([inString, outString])
    conValues = np.einsum(conString, *[coreDict[key].values for key in coreDict])

    return gc.NumpyTensorCore(conValues, outColors, name=name)


## VariablesList optimization:
# def optimize_variablesList(coreColorDict, colorDimDict, globallyOpenColors):
#    candidateColors = All Colors
#    variablesList = []
#    scoresDict = get_scoresDict(coreColorDict, colorDimDict, globallyOpenColors, candidateColors)

# def get_scoresDict(coreColorDict, colorDimDict, globallyOpenColors, candidateColors):
#    return { candidateColor: get_score(coreColorDict, colorDimDict, globallyOpenColors, candidateColor) for candidateColor in candidateColors}

# def get_score(coreColorDict, colorDimDict, globallyOpenColors, candidateColor):
#    score = 0
#    for coreKey in coreColorDict:
#        pass

if __name__ == "__main__":
    from tnreason.logic import coordinate_calculus as cc

    coreDict = {
        "c1": cc.CoordinateCore(np.random.binomial(10, 0.5, size=(3, 2)), ["a", "b"]),
        "c2": cc.CoordinateCore(np.random.binomial(20, 0.8, size=(3, 2, 5)), ["a", "b", "c"]),
        "c3": cc.CoordinateCore(np.random.binomial(20, 0.8, size=(3, 2, 5)), ["a", "b", "c"]),
        "c4": cc.CoordinateCore(np.random.binomial(20, 0.8, size=(3, 2, 5)), ["a", "b", "c"])
    }

    contractor = VariableEliminator(coreDict, variablesList=[["c"], ["b"]])
    contractor.contract()

    print(contractor.coreDict["contracted"].colors)
