from tnreason.logic import coordinate_calculus as cc
from tnreason.logic import expression_utils as eu

from tnreason.contraction import bc_contraction_generation as bcg
from tnreason.contraction import core_contractor as coc

import numpy as np


class FormulaTensor:
    '''
    expression:
    formulaKey: For distinction , typically passed by TensorRepresentation as dict key
    headType: Whether \weight\ftensor (truthEvaluation, default) or \expof{\weight\ftensor} (expFactor) or \partial\expof{\weight\ftensor} (diffExpFactor) is created
    weight: Factor on the formulaTensor
    '''

    def __init__(self, expression, formulaKey=None, headType="truthEvaluation", weight=1):
        if formulaKey is not None:
            self.formulaKey = formulaKey
        else:
            self.formulaKey = str(expression)
        self.expression = expression
        self.atoms = eu.get_variables(expression)

        ## Build the Cores
        self.create_subExpressionCores()
        self.set_head(headType=headType, weight=weight)

    def create_subExpressionCores(self):
        self.subExpressionCoresDict = bcg.create_formulaProcedure(self.expression, str(self.formulaKey))

    def set_head(self, headType, weight=1):
        if headType == "truthEvaluation":
            headValues = np.zeros(shape=(2))
            headValues[1] = weight
        elif headType == "expFactor":
            headValues = create_expFactor_values(weight, False)
        elif headType == "diffExpFactor":
            headValues = create_expFactor_values(weight, True)
        else:
            raise ValueError("Headtype {} not understood!".format(headType))
        self.headCore = cc.CoordinateCore(headValues, [self.formulaKey + "_" + str(self.expression)])

    def infer_on_evidenceDict(self, evidenceDict):
        evidenceCoresDict = {}
        for atomKey in evidenceDict:
            truthValues = np.zeros(shape=(2))
            if bool(evidenceDict[atomKey]):
                truthValues[1] = 1
            else:
                truthValues[0] = 1
            evidenceCoresDict[atomKey + "_evidence"] = cc.CoordinateCore(truthValues, [atomKey], atomKey + "_evidence")
        return coc.CoreContractor(
            {**self.subExpressionCoresDict, self.formulaKey + "_head": self.headCore, **evidenceCoresDict},
            openColors=[atomKey for atomKey in self.atoms if atomKey not in evidenceDict]).contract()

    def get_all_cores(self):
        return {**self.subExpressionCoresDict, self.formulaKey+"_head": self.headCore}

class SuperposedFormulaTensor:
    ## Shall be the central object to be optimized during MLE
    # Gradient is just the omission of the respective parameterCore

    def __init__(self, skeletonExpression, candidatesDict, parameterCoresDict=None):
        self.skeletonExpression = skeletonExpression
        self.candidatesDict = candidatesDict

        self.parameterCoresDict = parameterCoresDict  # former variableCoresDict
        self.create_selectorCoresDict()
        self.create_skeletonCoreDict()

        self.dataCoresDict = {}

    def set_parameterCoresDict(self, parameterCoresDict):
        self.parameterCoresDict = parameterCoresDict

    def random_initialize_parameterCoresDict(self):
        for coreKey in self.parameterCoresDict:
            self.parameterCoresDict[coreKey].values = np.random.random(size=self.parameterCoresDict[coreKey].values.shape)

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

    ## All without Datacores!
    def get_all_fTensor_cores(self, parameterExceptionKeys=[]):
        return {**{key: self.parameterCoresDict[key] for key in self.parameterCoresDict if
                   key not in parameterExceptionKeys},
                **self.selectorCoresDict,
                **self.skeletonCoresDict}


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
    fTensor = FormulaTensor(expression, "f1", headType="truthEvaluation", weight=1)
    print(fTensor.infer_on_evidenceDict({"A1": 1}).values)
    cv.draw_contractionDiagram({**fTensor.subExpressionCoresDict, "head": fTensor.headCore})
