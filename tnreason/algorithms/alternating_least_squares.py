from tnreason import engine
from tnreason import encoding

import numpy as np


class ALS:
    """
    Implements the alternating least squares
        * networkCores: Main tensor network to be optimized
        * importanceList: List of tuples containing
            - tensor network specifying a loss by contraction
            - weight specifying the importance in the loss
        * importanceColors: Specifying the shared colors of networkCores and importanceList networks
        * targetCores: Specifying the fitting target after contraction
        * trivialKeys: Specifying cores of singe coordinates, which contribute only factors
    """

    def __init__(self, networkCores, importanceColors=[], importanceList=[({}, 1)],
                 contractionMethod=engine.defaultContractionMethod, targetCores={}):
        self.networkCores = networkCores

        self.importanceColors = importanceColors
        self.importanceList = importanceList
        self.contractionMethod = contractionMethod

        self.targetCores = targetCores

        self.trivialKeys = []  # Keys with single position, trivial in the sense that they will not be updated

    def random_initialize(self, updateKeys, shapesDict={}, colorsDict={}):
        for updateKey in updateKeys:
            if updateKey in self.networkCores:
                upShape = self.networkCores[updateKey].values.shape
                upColors = self.networkCores[updateKey].colors
                self.networkCores.pop(updateKey)
            else:
                upShape = shapesDict[updateKey]
                upColors = colorsDict[updateKey]
            if np.prod(upShape) > 1:
                self.networkCores[updateKey] = encoding.create_random_core(updateKey, upShape, upColors,
                                                                           randomEngine="NumpyUniform")
            else:
                self.trivialKeys.append(updateKey)
                self.networkCores[updateKey] = encoding.create_trivial_core(updateKey, upShape, upColors)

    def alternating_optimization(self, updateKeys, sweepNum=10, computeResiduum=False):
        updateKeys = [key for key in updateKeys if key not in self.trivialKeys]
        if computeResiduum:
            residua = np.empty((sweepNum, len(updateKeys)))
        for sweep in range(sweepNum):
            for i, updateKey in enumerate(updateKeys):
                self.optimize_core(updateKey)
                if computeResiduum:
                    residua[sweep, i] = self.compute_residuum()
        if computeResiduum:
            return residua

    def get_color_argmax(self, updateKeys):
        # ! Only working for vectors #
        return {self.networkCores[key].colors[0]: np.argmax(np.abs(self.networkCores[key].values)) for key in
                updateKeys}

    def optimize_core(self, updateKey):
        ## Trivialize the core to be updated (serving as a placeholder)
        tbUpdated = self.networkCores.pop(updateKey)
        self.networkCores[updateKey] = encoding.create_trivial_core(updateKey, tbUpdated.values.shape, tbUpdated.colors)

        ## Compute flattened operator and target
        updateColors = tbUpdated.colors
        conOperator = self.compute_conOperator(updateColors, importanceCores=self.importanceList[0][0],
                                               weight=self.importanceList[0][1])
        conTarget = self.compute_conTarget(updateColors, importanceCores=self.importanceList[0][0],
                                           weight=self.importanceList[0][1])
        for importanceCores, weight in self.importanceList[1:]:
            conOperator = conOperator.sum_with(
                self.compute_conOperator(updateColors, importanceCores, weight))
            conTarget = conTarget.sum_with(self.compute_conTarget(updateColors, importanceCores, weight))

        resultDim = int(np.prod(conTarget.values.shape))
        conOperator.reorder_colors(conTarget.colors + [color + "_out" for color in conTarget.colors])
        flattenedOperator = conOperator.values.reshape(resultDim, resultDim)
        flattenedTarget = conTarget.values.flatten()

        ## Update the core by solution of least squares problem
        solution, res, rank, s = np.linalg.lstsq(flattenedOperator, flattenedTarget, rcond=None)
        self.networkCores[updateKey] = engine.get_core()(solution.reshape(tbUpdated.values.shape), updateColors,
                                                         updateKey)

    def compute_conOperator(self, updateColors, importanceCores={}, weight=1):
        return engine.contract(method=self.contractionMethod,
                               coreDict={
                                   **importanceCores,
                                   **self.networkCores,
                                   **copy_cores(self.networkCores, "_out", self.importanceColors)
                               }, openColors=updateColors + [updateColor + "_out" for updateColor in
                                                             updateColors]).multiply(weight)

    def compute_conTarget(self, updateColors, importanceCores={}, weight=1):
        return engine.contract(method=self.contractionMethod,
                               coreDict={
                                   **importanceCores,
                                   **self.networkCores,
                                   **self.targetCores,
                               }, openColors=updateColors).multiply(weight)

    def compute_residuum(self):
        prediction = engine.contract(method=self.contractionMethod,
                                     coreDict=self.networkCores,
                                     openColors=self.importanceColors)
        target = engine.contract(method=self.contractionMethod,
                                 coreDict=self.targetCores,
                                 openColors=self.importanceColors)
        prediction.reorder_colors(target.colors)
        ## Not using the weightings by the importanceList!
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
