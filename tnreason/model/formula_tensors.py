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

    def create_selectorCoresDict(self, candidatesDict):
        ## incolors: placeHolderKey
        ## outcolors: placeHolderKey + "_" + atomKey
        self.selectorCoresDict = {}
        for placeHolderKey in candidatesDict:
            for i, atomKey in enumerate(candidatesDict[placeHolderKey]):
                coreValues = np.ones(shape=(len(candidatesDict[placeHolderKey]), 2))
                coreValues[i, 0] = 0
                self.selectorCoresDict[placeHolderKey + "_" + atomKey+"_selector"] = cc.CoordinateCore(
                    coreValues, [placeHolderKey, placeHolderKey+"_"+atomKey], placeHolderKey + "_" + atomKey + "_selector")

    def create_skeletonCoreDict(self, skeletonExpression, candidatesDict):
        ## incolors: placeHolderKey + "_" + atomKey
        ## outcolors: atomKey
        self.skeletonCoresDict, self.atoms = skeleton_recursion(skeletonExpression, candidatesDict)
        for atomKey in self.atoms:
            self.skeletonCoresDict[atomKey+"_skeletonHeadCore"] = create_deltaCore([str(skeletonExpression)+"_"+atomKey, atomKey], atomKey+"_skeletonHeadCore")


    ## WorldCoresDict Generation: CandidatesDict required for interpretation of the
    # candidatesDict gives interpretation of placeholder axes
    def create_atomDataCores(self, sampleDf):
        self.dataCoresDict = {
            atomKey+"_data": dataCore_from_sampleDf(sampleDf, atomKey)
            for atomKey in self.atoms
        }


def dataCore_from_sampleDf(sampleDf, atomKey):
    if atomKey not in sampleDf.keys():
        raise ValueError
    dfEntries = sampleDf[atomKey].values
    dataNum = dfEntries.shape[0]
    values = np.zeros(shape=(dataNum, 2))
    for i in range(dataNum):
        if dfEntries[i] == 0:
            values[i, 0] = 1
        else:
            values[i, 1] = 1
    return cc.CoordinateCore(values, ["j", atomKey])

def skeleton_recursion(headExpression, candidatesDict):
    print(headExpression)
    if type(headExpression) == str:
        return {}, candidatesDict[headExpression]
    elif headExpression[0] == "not":
        if type(headExpression[1]) == str:
            return create_negationCoreDict(candidatesDict[headExpression[1]], inprefix=str(headExpression[1])+"_", outprefix=str(headExpression)+"_"), candidatesDict[headExpression[1]]
        else:
            #atoms = eu.get_variables(headExpression[1])
            skeletonCoreDict, atoms = skeleton_recursion(headExpression[1], candidatesDict)
            return {**skeletonCoreDict,
                **create_negationCoreDict(atoms, inprefix = str(headExpression[1])+"_", outprefix = str(headExpression))+"_"}, atoms
    elif headExpression[1] == "and":
        if type(headExpression[0]) == str:
            leftSkeletonCoreDict = {headExpression[0]+"_"+atomKey +"_l" : create_deltaCore(colors=[headExpression[0]+"_"+atomKey, str(headExpression)+"_"+atomKey])
                                    for atomKey in candidatesDict[headExpression[0]]}
            leftatoms = candidatesDict[headExpression[0]]
        else:
            leftSkeletonCoreDict, leftatoms = skeleton_recursion(headExpression[0], candidatesDict)

            leftSkeletonCoreDict = {**leftSkeletonCoreDict,
                                     **{str(headExpression[0])+"_"+atomKey+"_lPass": create_deltaCore([str(headExpression[0])+"_"+atomKey,str(headExpression)+"_"+atomKey])
                                     for atomKey in leftatoms}
                                     }
        if type(headExpression[2]) == str:
            rightSkeletonCoreDict = {headExpression[2]+"_"+atomKey +"_r": create_deltaCore(colors=[headExpression[2]+"_"+atomKey, str(headExpression)+"_"+atomKey])
                                    for atomKey in candidatesDict[headExpression[2]]}
            rightatoms = candidatesDict[headExpression[2]]
        else:
            rightSkeletonCoreDict, rightatoms = skeleton_recursion(headExpression[2], candidatesDict)
            rightSkeletonCoreDict = {**rightSkeletonCoreDict,
                                     **{str(headExpression[2])+"_"+atomKey+"_rPass": create_deltaCore([str(headExpression[2])+"_"+atomKey,str(headExpression)+"_"+atomKey])
                                     for atomKey in rightatoms}
                                     }
        return {**leftSkeletonCoreDict, **rightSkeletonCoreDict}, leftatoms + rightatoms

negationMatrix = np.zeros(shape=(2, 2))
negationMatrix[0, 1] = 1
negationMatrix[1, 0] = 1


def create_negationCoreDict(atoms, inprefix, outprefix):
    negationCoreDict = {}
    for atomKey in atoms:
        negationCoreDict[outprefix + "_"+atomKey + "_neg"] = cc.CoordinateCore(negationMatrix,
                                                                        [inprefix + atomKey, outprefix + atomKey], outprefix + atomKey + "_neg")
    return negationCoreDict

def create_deltaCore(colors, name=""):
    values = np.zeros(shape=[2 for i in range(len(colors))])
    values[tuple(0 for color in colors)] = 1
    values[tuple(1 for color in colors)] = 1
    return cc.CoordinateCore(values, colors, name)


if __name__ == "__main__":

    from tnreason.contraction import contraction_visualization as cv

    skeletonExpression = ["P1","and",["not","P2"]]
    candidatesDict = {"P1": ["A1", "A2"], "P2": ["A2"]}

    supFtensor = SuperposedFormulaTensor()
    supFtensor.set_parameterCoresDict({
        "vCore1": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P1", "H1"]),
        "vCore2": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P2", "H1"]),
    })
    supFtensor.create_selectorCoresDict(candidatesDict)
    supFtensor.create_skeletonCoreDict(skeletonExpression, candidatesDict)

    learnedFormulaDict = {
        "f0": ["A1", 10],
        "f1": [["not", ["A2", "and", "A3"]], 5],
        "f2": ["A2", 2]
    }
    import tnreason.model.generate_test_data as gtd
    sampleDf = gtd.generate_sampleDf(learnedFormulaDict, 100)
    supFtensor.create_atomDataCores(sampleDf)

    cv.draw_contractionDiagram({**supFtensor.parameterCoresDict,
                                **supFtensor.skeletonCoresDict,
                                **supFtensor.selectorCoresDict,
                                **supFtensor.dataCoresDict})