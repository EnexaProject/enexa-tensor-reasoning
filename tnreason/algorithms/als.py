from tnreason import engine
from tnreason import encoding

import numpy as np

defaultContractionMethod = "PgmpyVariableEliminator"
defaultCoreType = "NumpyTensorCore"


class ALS:
    def __init__(self, networkCores, importanceColors=[], importanceList=[({}, 1)],
                 contractionMethod=defaultContractionMethod, targetCores={}):
        self.networkCores = networkCores

        self.importanceColors = importanceColors
        self.importanceList = importanceList
        self.contractionMethod = contractionMethod

        self.targetCores = targetCores

    def random_initialize(self, updateKeys, shapesDict={}, colorsDict={}, coreType=defaultCoreType):
        for updateKey in updateKeys:
            if updateKey in self.networkCores:
                upShape = self.networkCores[updateKey].values.shape
                upColors = self.networkCores[updateKey].colors
                self.networkCores.pop(updateKey)
            else:
                upShape = shapesDict[updateKey]
                upColors = colorsDict[updateKey]
            self.networkCores[updateKey] = engine.get_core(coreType)(np.random.random(size=upShape), upColors,
                                                                     updateKey)

    def alternating_optimization(self, updateKeys, sweepNum=10, computeResiduum=False):
        if computeResiduum:
            residua = np.empty((sweepNum, len(updateKeys)))
        for sweep in range(sweepNum):
            for i, updateKey in enumerate(updateKeys):
                self.optimize_core(updateKey)
                if computeResiduum:
                    residua[sweep, i] = self.compute_residuum()
        if computeResiduum:
            return residua

    def compute_conOperator(self, updateColors, updateShape, importanceCores={}, weight=1):
        trivialCores = encoding.get_trivial_cores(
            updateColors + [updateColor + "_out" for updateColor in updateColors],
            varDimDict={**{color: updateShape[i] for i, color in enumerate(updateColors)},
                        **{color + "_out": updateShape[i] for i, color in enumerate(updateColors)}
                        },
            coreType=defaultCoreType
        )

        return engine.contract(method=self.contractionMethod,
                               coreDict={
                                   **importanceCores,
                                   **self.networkCores,
                                   **copy_cores(self.networkCores, "_out", self.importanceColors),
                                   **trivialCores
                               }, openColors=updateColors + [updateColor + "_out" for updateColor in
                                                             updateColors]).multiply(weight)

    def compute_conTarget(self, updateColors, updateShape, importanceCores={}, weight=1):
        return engine.contract(method=self.contractionMethod,
                               coreDict={
                                   **importanceCores,
                                   **self.networkCores,
                                   **self.targetCores,
                                   **encoding.get_trivial_cores(updateColors,
                                                                varDimDict={color: updateShape[i] for i, color in
                                                                            enumerate(updateColors)})
                               }, openColors=updateColors).multiply(weight)

    def optimize_core(self, updateKey, coreType=defaultCoreType):
        tbUpdated = self.networkCores.pop(updateKey)
        updateColors = tbUpdated.colors
        updateShape = tbUpdated.values.shape

        conOperator = self.compute_conOperator(updateColors, updateShape, importanceCores=self.importanceList[0][0],
                                               weight=self.importanceList[0][1])
        conTarget = self.compute_conTarget(updateColors, updateShape, importanceCores=self.importanceList[0][0],
                                           weight=self.importanceList[0][1])
        for importanceCores, weight in self.importanceList[1:]:
            conOperator = conOperator.sum_with(
                self.compute_conOperator(updateColors, updateShape, importanceCores, weight))
            conTarget = conTarget.sum_with(self.compute_conTarget(updateColors, updateShape, importanceCores, weight))

        resultDim = int(np.prod(conTarget.values.shape))
        conOperator.reorder_colors(conTarget.colors + [color + "_out" for color in conTarget.colors])

        flattenedOperator = conOperator.values.reshape(resultDim, resultDim)
        flattenedTarget = conTarget.values.flatten()

        solution, res, rank, s = np.linalg.lstsq(flattenedOperator, flattenedTarget)

        self.networkCores[updateKey] = engine.get_core(coreType)(solution.reshape(updateShape), updateColors, updateKey)

    def compute_residuum(self):
        prediction = engine.contract(method=self.contractionMethod, coreDict=
        self.networkCores, openColors=self.importanceColors
                                     )
        target = engine.contract(method=self.contractionMethod, coreDict=
        self.targetCores, openColors=self.importanceColors
                                 )
        prediction.reorder_colors(target.colors)
        return np.linalg.norm(prediction.values - target.values)


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
