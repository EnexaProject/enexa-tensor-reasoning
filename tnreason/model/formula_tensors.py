from tnreason.logic import coordinate_calculus as cc
from tnreason.logic import expression_utils as eu

from tnreason.contraction import layers as lay

import numpy as np


class SuperposedFormulaTensor:
    ## Shall be the central object to be optimized during MLE
    # Gradient is just the omission of the respective parameterCore

    def __init__(self):
        self.parameterCoresDict = {}  # former variableCoresDict
        self.worldCoresDict = {}  # former fixedCoresDict / atomSelectorDict
        self.skeletonCoresDict = {}  # new from lcg

    def set_parameterCoresDict(self, parameterCoresDict):
        self.parameterCoresDict = parameterCoresDict

    def create_skeletonCoreDict(self, skeletonExpression, placeHolderShapesDict, placeHolderColorsDict):
        # placeHolderDicts -> shapes and colors specify how skeletonTN looks like at each placeholder
        pass

    ## WorldCoresDict Generation: CandidatesDict required for interpretation of the
    # candidatesDict gives interpretation of placeholder axes
    def create_worldCoresDict_from_sampleDf(self, candidatesDict, sampleDf):
        self.worldCoresDict = {
            placeHolderKey: cc.CoordinateCore(sampleDf[candidatesDict[placeHolderKey]].values,
                                              candidatesDict[placeHolderKey], "sampleWorlds_" + placeHolderKey) for
            placeHolderKey in candidatesDict
        }

    def create_worldCoresDict_from_enumeration(self, candidatesDict):
        self.worldCoresDict = {}
        for placeHolderKey in candidatesDict:
            for i, atomKey in enumerate(candidatesDict[placeHolderKey]):
                coreValues = np.ones(shape=(len(candidatesDict[placeHolderKey]), 2))
                coreValues[i, 0] = 0
                self.worldCoresDict["enumeratedWorlds_" + placeHolderKey + "_" + atomKey] = cc.CoordinateCore(
                    coreValues, [placeHolderKey, atomKey], "enumeratedWorlds_" + placeHolderKey + "_" + atomKey)





def generate_negation_layer(skeletonExpression, shapesDict, sampleNum):
    variableColors = eu.get_variables(skeletonExpression)
    variableShapes = [shapesDict[color] for color in variableColors]

    directCore = cc.CoordinateCore(np.eye(sampleNum), [str(skeletonExpression[0]), str(skeletonExpression)])
    constantCore = cc.CoordinateCore(np.ones(shape=variableShapes+[sampleNum]), variableColors + [str(skeletonExpression)])

    return lay.Layer({str(skeletonExpression)+"_"+"constant": constantCore,
                      str(skeletonExpression)+"_"+"direct": directCore},
                     {str(skeletonExpression) + "_" + "constant": 1,
                      str(skeletonExpression) + "_" + "direct": -1})

def create_skeletonLayerDict(skeletonExpression, shapesDict):
    addCoreKey = str(skeletonExpression)
    if type(skeletonExpression)==str:
        return {}
    elif skeletonExpression[0] == "not":
        layerDict = create_skeletonLayerDict(skeletonExpression[1], shapesDict)
        layerDict[addCoreKey] = generate_negation_layer(skeletonExpression, candidatesDict)


if __name__ == "__main__":
    candidatesDict = {"P1" : ["A1","A2","A3"], "P2" : ["A4","A5"]}

    supFtensor = SuperposedFormulaTensor()
    supFtensor.create_worldCoresDict_from_enumeration(candidatesDict)

    from tnreason.contraction import contraction_visualization as cv
    cv.draw_contractionDiagram(supFtensor.worldCoresDict,fontsize=5)