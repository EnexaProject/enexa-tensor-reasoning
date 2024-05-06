from tnreason import engine
from tnreason import encoding

import numpy as np

momentCoreSuffix = encoding.headCoreSuffix
targetCoreSuffix = "_targetCore"

class MomentMatcher:
    def __init__(self, networkCores, targetCores):
        self.networkCores = networkCores

        self.targetCores = {targetCores[key].colors[0] + targetCoreSuffix: targetCores[key] for key in targetCores}

        self.updateDimDict = {self.targetCores[key].colors[0]: self.targetCores[key].values.shape[0] for key in
                              self.targetCores}

    def ones_initialization(self):
        """
        varDimDict: Dictionary with keys the colors of the moments and the shape the dimension of the axis
        """
        self.networkCores.update(
            encoding.create_emptyCoresDict(list(self.updateDimDict.keys()), varDimDict=self.updateDimDict,
                                           suffix=momentCoreSuffix)
        )

    def matching_step(self, updateColor):
        self.networkCores.pop(updateColor + momentCoreSuffix)

        self.networkCores[updateColor + momentCoreSuffix] = engine.get_core()(
            values=solve_moment_equation(
                satVect=engine.contract(coreDict=self.networkCores, openColors=[updateColor]).values,
                empVect=self.targetCores[updateColor + targetCoreSuffix].values
            ), colors=[updateColor], name=updateColor + momentCoreSuffix)

        print(self.networkCores[updateColor + momentCoreSuffix].values)

    def alternating_matching(self, sweepNum = 10, updateColors=None):
        if updateColors is None:
            updateColors = list(self.updateDimDict.keys())
        for sweep in range(sweepNum):
            for updateColor in updateColors:
                self.matching_step(updateColor)

def find_common_nonzero(vect1, vec2):
    for i in range(vect1.shape[0]):
        if vect1[i] > 0 and vec2[i] > 0:
            return i
    else:
        return -1


def solve_moment_equation(satVect, empVect):
    solVect = np.ones(shape=satVect.shape)

    refPos = find_common_nonzero(satVect, empVect)
    if refPos == -1:
        print("Warning: Moments cannot be matched!")
        return solVect

    for i in range(satVect.shape[0]):
        solVect[i] = (satVect[refPos] / empVect[refPos]) * (empVect[i] / satVect[i])

    return solVect
