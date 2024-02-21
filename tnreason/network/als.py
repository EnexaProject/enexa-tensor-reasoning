from tnreason import contraction
from tnreason import tensor

from tnreason.tensor import model_cores as mcore

import numpy as np

defaultContractionMethod = "PgmpyVariableEliminator"
defaultCoreType = "NumpyTensorCore"


class ALS:
    def __init__(self, networkCores, targetCores, openTargetColors, contractionMethod=defaultContractionMethod):
        self.networkCores = networkCores
        self.targetCores = targetCores
        self.openTargetColors = openTargetColors
        self.contractionMethod = contractionMethod

    def random_initialize(self, updateKeys, shapesDict={}, colorsDict={}, coreType=defaultCoreType):
        for updateKey in updateKeys:
            if updateKey in self.networkCores:
                upShape = self.networkCores[updateKey].values.shape
                upColors = self.networkCores[updateKey].colors
                self.networkCores.pop(updateKey)
            else:
                upShape = shapesDict[updateKey]
                upColors = colorsDict[updateKey]
            self.networkCores[updateKey] = tensor.get_core(coreType)(np.random.random(size=upShape), upColors,
                                                                     updateKey)

    def alternating_optimization(self, updateKeys, sweepNum=10):
        for sweep in range(sweepNum):
            for updateKey in updateKeys:
                self.optimize_core(updateKey)

    def optimize_core(self, updateKey, coreType=defaultCoreType):
        tbUpdated = self.networkCores.pop(updateKey)
        updateColors = tbUpdated.colors
        updateShape = tbUpdated.values.shape

        trivialCores = mcore.create_emptyCoresDict(
            updateColors + [updateColor + "_out" for updateColor in updateColors],
            varDimDict={**{color: updateShape[i] for i, color in enumerate(updateColors)},
                        **{color + "_out": updateShape[i] for i, color in enumerate(updateColors)}
                        },
            coreType=defaultCoreType
        )

        conOperator = contraction.get_contractor(self.contractionMethod)({
            **self.networkCores,
            **copy_cores(self.networkCores, "_out", self.openTargetColors),
            **trivialCores
        }, openColors=updateColors + [updateColor + "_out" for updateColor in updateColors]).contract()

        conTarget = contraction.get_contractor(self.contractionMethod)({
            **self.networkCores,
            **self.targetCores,
            **mcore.create_emptyCoresDict(
                updateColors,
                varDimDict={color: updateShape[i] for i, color in enumerate(updateColors)}),
        }, openColors=updateColors).contract()

        resultDim = int(np.prod(conTarget.values.shape))
        conOperator.reorder_colors(conTarget.colors + [color + "_out" for color in conTarget.colors])

        flattenedOperator = conOperator.values.reshape(resultDim, resultDim)
        flattenedTarget = conTarget.values.flatten()

        solution, res, rank, s = np.linalg.lstsq(flattenedOperator, flattenedTarget)

        self.networkCores[updateKey] = tensor.get_core(coreType)(solution.reshape(updateShape), updateColors, updateKey)


def copy_cores(coreDict, suffix, exceptionColors):
    returnDict = {}
    for key in coreDict:
        core = coreDict[key].clone()
        newColors = core.colors
        for i, color in enumerate(newColors):
            if color not in exceptionColors:
                newColors[i] = color + suffix
        core.colors = newColors
        returnDict[key + suffix] = core
    return returnDict


def change_color_in_coredict(coreDict, colorReplaceDict, replaceSuffix="_out"):
    returnDict = {}
    for key in coreDict.copy():
        core = coreDict[key].clone()
        newColors = core.colors
        for i, color in enumerate(newColors):
            if color in colorReplaceDict:
                newColors[i] = colorReplaceDict[color]
        core.colors = newColors
        returnDict[key + replaceSuffix] = core
    return returnDict
