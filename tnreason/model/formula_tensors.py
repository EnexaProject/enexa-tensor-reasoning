from tnreason.logic import coordinate_calculus as cc

from tnreason.contraction import bc_contraction_generation as bcg

import numpy as np


class FormulaTensor:
    def __init__(self, expression, key=None, headType="truthEvaluation", weight=0):
        if key is not None:
            self.formulaKey = key
        else:
            self.formulaKey = str(expression)
        self.expression = expression
        self.create_subExpressionCores()

        self.set_head(headType=headType, weight=weight)

    def create_subExpressionCores(self):
        self.subExpressionCoresDict = bcg.create_formulaProcedure(self.expression, str(self.formulaKey))

    def set_head(self, headType, weight=0):
        if headType == "truthEvaluation":
            headValues = np.zeros(shape=(2))
            headValues[1] = 1
        elif headType == "expFactor":
            headValues = create_expFactor_values(weight, False)
        elif headType == "diffExpFactor":
            headValues = create_expFactor_values(weight, True)
        self.headCore = cc.CoordinateCore(headValues, [self.formulaKey + "_" + str(self.expression)])


class SuperposedFormulaTensor:
    ## Shall be the central object to be optimized during MLE
    # Gradient is just the omission of the respective parameterCore

    def __init__(self, skeletonExpression, candidatesDict, parameterCoresDict=None):
        self.skeletonExpression = skeletonExpression
        self.candidatesDict = candidatesDict

        self.parameterCoresDict = parameterCoresDict  # former variableCoresDict
        self.create_selectorCoresDict()
        self.create_skeletonCoreDict()

    def set_parameterCoresDict(self, parameterCoresDict):
        self.parameterCoresDict = parameterCoresDict

    def create_selectorCoresDict(self):
        ## incolors: placeHolderKey
        ## outcolors: placeHolderKey + "_" + atomKey
        self.selectorCoresDict = {}
        for placeHolderKey in self.candidatesDict:
            for i, atomKey in enumerate(self.candidatesDict[placeHolderKey]):
                coreValues = np.ones(shape=(len(self.candidatesDict[placeHolderKey]), 2))
                coreValues[i, 0] = 0
                self.selectorCoresDict[placeHolderKey + "_" + atomKey + "_selector"] = cc.CoordinateCore(
                    coreValues, [placeHolderKey, placeHolderKey + "_" + atomKey],
                    placeHolderKey + "_" + atomKey + "_selector")

    def create_skeletonCoreDict(self):
        ## incolors: placeHolderKey + "_" + atomKey
        ## outcolors: atomKey
        self.skeletonCoresDict, self.atoms = skeleton_recursion(self.skeletonExpression, self.candidatesDict)
        for atomKey in self.atoms:
            self.skeletonCoresDict[atomKey + "_skeletonHeadCore"] = create_deltaCore(
                [str(self.skeletonExpression) + "_" + atomKey, atomKey], atomKey + "_skeletonHeadCore")

    ## WorldCoresDict Generation: CandidatesDict required for interpretation of the
    # candidatesDict gives interpretation of placeholder axes
    def create_atomDataCores(self, sampleDf):
        self.dataCoresDict = {
            atomKey + "_data": dataCore_from_sampleDf(sampleDf, atomKey)
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
    if type(headExpression) == str:
        return {}, candidatesDict[headExpression]
    elif headExpression[0] == "not":
        if type(headExpression[1]) == str:
            return create_negationCoreDict(candidatesDict[headExpression[1]], inprefix=str(headExpression[1]) + "_",
                                           outprefix=str(headExpression) + "_"), candidatesDict[headExpression[1]]
        else:
            skeletonCoreDict, atoms = skeleton_recursion(headExpression[1], candidatesDict)
            return {**skeletonCoreDict,
                    **create_negationCoreDict(atoms, inprefix=str(headExpression[1]) + "_",
                                              outprefix=str(headExpression)) + "_"}, atoms
    elif headExpression[1] == "and":
        if type(headExpression[0]) == str:
            leftSkeletonCoreDict = {headExpression[0] + "_" + atomKey + "_l": create_deltaCore(
                colors=[headExpression[0] + "_" + atomKey, str(headExpression) + "_" + atomKey])
                                    for atomKey in candidatesDict[headExpression[0]]}
            leftatoms = candidatesDict[headExpression[0]]
        else:
            leftSkeletonCoreDict, leftatoms = skeleton_recursion(headExpression[0], candidatesDict)

            leftSkeletonCoreDict = {**leftSkeletonCoreDict,
                                    **{str(headExpression[0]) + "_" + atomKey + "_lPass": create_deltaCore(
                                        [str(headExpression[0]) + "_" + atomKey, str(headExpression) + "_" + atomKey])
                                       for atomKey in leftatoms}
                                    }
        if type(headExpression[2]) == str:
            rightSkeletonCoreDict = {headExpression[2] + "_" + atomKey + "_r": create_deltaCore(
                colors=[headExpression[2] + "_" + atomKey, str(headExpression) + "_" + atomKey])
                                     for atomKey in candidatesDict[headExpression[2]]}
            rightatoms = candidatesDict[headExpression[2]]
        else:
            rightSkeletonCoreDict, rightatoms = skeleton_recursion(headExpression[2], candidatesDict)
            rightSkeletonCoreDict = {**rightSkeletonCoreDict,
                                     **{str(headExpression[2]) + "_" + atomKey + "_rPass": create_deltaCore(
                                         [str(headExpression[2]) + "_" + atomKey, str(headExpression) + "_" + atomKey])
                                        for atomKey in rightatoms}
                                     }
        return {**leftSkeletonCoreDict, **rightSkeletonCoreDict}, leftatoms + rightatoms


def create_negationCoreDict(atoms, inprefix, outprefix):
    negationMatrix = np.zeros(shape=(2, 2))
    negationMatrix[0, 1] = 1
    negationMatrix[1, 0] = 1

    negationCoreDict = {}
    for atomKey in atoms:
        negationCoreDict[outprefix + "_" + atomKey + "_neg"] = cc.CoordinateCore(negationMatrix,
                                                                                 [inprefix + atomKey,
                                                                                  outprefix + atomKey],
                                                                                 outprefix + atomKey + "_neg")
    return negationCoreDict


def create_deltaCore(colors, name=""):
    values = np.zeros(shape=[2 for i in range(len(colors))])
    values[tuple(0 for color in colors)] = 1
    values[tuple(1 for color in colors)] = 1
    return cc.CoordinateCore(values, colors, name)


def create_expFactor_values(weight, differentiated):
    values = np.zeros(shape=(2))
    if not differentiated:
        values[0] = 1
    values[1] = np.exp(weight)
    return values


if __name__ == "__main__":
    from tnreason.contraction import contraction_visualization as cv

    expression = ["A1", "and", ["not", "A2"]]
    fTensor = FormulaTensor(expression, "f1", headType="expFactor", weight=1)

    cv.draw_contractionDiagram({**fTensor.subExpressionCoresDict, "head": fTensor.headCore})
