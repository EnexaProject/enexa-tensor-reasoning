from tnreason import contraction
from tnreason import tensor

from tnreason import engine

from tnreason.tensor import model_cores as mcore

import numpy as np

defaultContractionMethod = "PgmpyVariableEliminator"
defaultCoreType = "NumpyTensorCore"


class Gibbs:
    def __init__(self, networkCores, openTargetColors=[], importanceList=[({}, 1)],
                 contractionMethod=defaultContractionMethod):
        self.networkCores = networkCores

        self.openTargetColors = openTargetColors
        self.importanceList = importanceList
        self.contractionMethod = contractionMethod

    def ones_initialization(self, updateKeys, shapesDict, colorsDict):
        for updateKey in updateKeys:
            upShape = shapesDict[updateKey]
            upColors = colorsDict[updateKey]
            self.networkCores[updateKey] = engine.get_core(defaultCoreType)(np.random.random(size=upShape), upColors,
                                                                            updateKey)

    def alternating_sampling(self, updateKeys, sweepNum=10, temperature=1, computeResiduum=False):
        positions = np.empty(shape=(sweepNum, len(updateKeys)))
        for sweep in range(sweepNum):
            for i, updateKey in enumerate(updateKeys):
                positions[sweep, i] = self.sample_core(updateKey, temperature=temperature)
        return positions

    def sample_core(self, updateKey, temperature):
        tbUpdated = self.networkCores.pop(updateKey)
        updateColors = tbUpdated.colors
        updateShape = tbUpdated.values.shape

        updateDistribution = engine.contract({**self.networkCores, **self.importanceList[0][0]}, updateColors,
                                             method=defaultContractionMethod).multiply(self.importanceList[0][1])
        updateDistribution.reorder_colors(updateColors)
        for importanceCores, weight in self.importanceList[1:]:
            updateDistribution = updateDistribution.sum_with(
                engine.contract({**self.networkCores, **importanceCores}, updateColors,
                                method=defaultContractionMethod).multiply(weight))
        flattened = updateDistribution.values.flatten()

        if np.sum(flattened) <= 0:
            localProb = np.ones(shape=flattened.shape) / flattened.shape[0]
            print("Vanishing prob for update of core {}!".format(updateKey))
        else:
            if temperature != 1:
                localProb = flattened ** temperature / (np.sum(flattened ** temperature))
            else:
                localProb = flattened / np.sum(flattened)

        newCore = np.zeros(shape=(np.prod(updateShape)))

        pos = np.where(np.random.multinomial(1, localProb) == 1)[0][0]
        newCore[pos] = 1
        self.networkCores[updateKey] = engine.get_core(defaultCoreType)(newCore, updateColors)
        return pos
