from tnreason.logic import coordinate_calculus as cc
from tnreason.logic import expression_utils as eu

from tnreason.contraction import core_contractor as coc
from tnreason.model import create_cores as crc

import numpy as np


class FormulaTensor:
    '''
    expression:
    formulaKey: For distinction , typically passed by TensorRepresentation as dict key
    headType: Whether \weight\ftensor (truthEvaluation, default) or \expof{\weight\ftensor} (expFactor) or \partial\expof{\weight\ftensor} (diffExpFactor) is created
    weight: Factor on the formulaTensor
    '''

    def __init__(self, expression, formulaKey=None, headType="weightedTruthEvaluation", weight=1):
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
        self.subExpressionCoresDict = crc.create_formulaProcedure(self.expression, str(self.formulaKey))

    def set_head(self, headType, weight=1):
        if headType == "truthEvaluation":
            headValues = np.zeros(shape=(2))
            headValues[1] = 1  # weight
        elif headType == "weightedTruthEvaluation":
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

    def get_cores(self):
        return {**self.subExpressionCoresDict, self.formulaKey + "_head": self.headCore}


class SuperposedFormulaTensor:
    ## Shall be the central object to be optimized during MLE
    # Gradient is just the omission of the respective parameterCore

    def __init__(self, skeletonExpression, candidatesDict, parameterCoresDict=None):
        self.skeletonExpression = skeletonExpression
        self.candidatesDict = candidatesDict

        if parameterCoresDict is not None:  # former variableCoresDict
            self.set_parameterCoresDict(parameterCoresDict)
        self.create_selectorCoresDict()
        self.create_skeletonCoresDict()

        self.dataCoresDict = {}

    def set_parameterCoresDict(self, parameterCoresDict):
        self.parameterCoresDict = parameterCoresDict
        ## Check for consistency with the candidatesDict
        check_colorShapes([parameterCoresDict],
                          knownShapesDict={key: len(self.candidatesDict[key]) for key in self.candidatesDict})

    def random_initialize_parameterCoresDict(self):
        for coreKey in self.parameterCoresDict:
            self.parameterCoresDict[coreKey].values = np.random.random(
                size=self.parameterCoresDict[coreKey].values.shape)

    def get_largest_weight_as_solutionMap(self):
        ## Choose formula Tensor by maximal entry
        contractedParameters = coc.CoreContractor(self.parameterCoresDict,
                                                  openColors=self.candidatesDict.keys()).contract()
        maxPos = np.argmax(contractedParameters.values)
        maxIndices = np.unravel_index(maxPos, contractedParameters.values.shape)
        solutionMap = {
            str(contractedParameters.colors[i]): self.candidatesDict[str(contractedParameters.colors[i])][maxIndices[i]]
            for i in range(len(maxIndices))}
        return solutionMap

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

    def create_skeletonCoresDict(self):
        ## incolors: placeHolderKey + "_" + atomKey
        ## outcolors: atomKey
        self.skeletonCoresDict, self.atoms = crc.skeleton_recursion(self.skeletonExpression, self.candidatesDict)
        for atomKey in self.atoms:
            self.skeletonCoresDict[atomKey + "_skeletonHeadCore"] = crc.create_deltaCore(
                [str(self.skeletonExpression) + "_" + atomKey, atomKey], atomKey + "_skeletonHeadCore")

    def get_cores(self, parameterExceptionKeys=[]):
        return {**{key: self.parameterCoresDict[key] for key in self.parameterCoresDict if
                   key not in parameterExceptionKeys},
                **self.selectorCoresDict,
                **self.skeletonCoresDict}


class DataTensor:
    def __init__(self, sampleDf, atoms=None):
        if atoms is None:
            self.atoms = sampleDf.columns
        else:
            self.atoms = atoms
        self.dataNum = sampleDf.values.shape[0]
        self.dataCores = {
            atomKey + "_data": crc.dataCore_from_sampleDf(sampleDf, atomKey)
            for atomKey in self.atoms
        }

    def get_cores(self):
        return self.dataCores

    def compute_shannon_entropy(self):
        contractedData = coc.CoreContractor(self.dataCores,
                                            openColors=self.atoms).contract().values.flatten() / self.dataNum
        logContractedData = np.log(np.copy(contractedData))
        logContractedData[logContractedData < -1e308] = 0
        return -np.dot(logContractedData, contractedData)


## Check whether the colors in all coreDicts match wrt each other and the knownShapesDict
def check_colorShapes(coresDicts, knownShapesDict={}):
    for coresDict in coresDicts:
        for coreKey in coresDict:
            for i, color in enumerate(coresDict[coreKey].colors):
                coreColorShape = coresDict[coreKey].values.shape[i]
                if color not in knownShapesDict:
                    knownShapesDict[color] = coreColorShape
                else:
                    if knownShapesDict[color] != coreColorShape:
                        raise ValueError("Core {} has unexpected shape of color {}.".format(coreKey, color))





# ## When Placeholders in Expression (Ftensor)
# def skeleton_recursion(headExpression, candidatesDict):
#     if type(headExpression) == str:
#         return {}, candidatesDict[headExpression]
#     elif headExpression[0] == "not":
#         if type(headExpression[1]) == str:
#             return create_negationCoreDict(candidatesDict[headExpression[1]], inprefix=str(headExpression[1]) + "_",
#                                            outprefix=str(headExpression) + "_"), candidatesDict[headExpression[1]]
#         else:
#             skeletonCoresDict, atoms = skeleton_recursion(headExpression[1], candidatesDict)
#             return {**skeletonCoresDict,
#                     **create_negationCoreDict(atoms, inprefix=str(headExpression[1]) + "_",
#                                               outprefix=str(headExpression)) + "_"}, atoms
#     elif headExpression[1] == "and":
#         if type(headExpression[0]) == str:
#             leftskeletonCoresDict = {headExpression[0] + "_" + atomKey + "_l": create_deltaCore(
#                 colors=[headExpression[0] + "_" + atomKey, str(headExpression) + "_" + atomKey])
#                 for atomKey in candidatesDict[headExpression[0]]}
#             leftatoms = candidatesDict[headExpression[0]]
#         else:
#             leftskeletonCoresDict, leftatoms = skeleton_recursion(headExpression[0], candidatesDict)
#
#             leftskeletonCoresDict = {**leftskeletonCoresDict,
#                                     **{str(headExpression[0]) + "_" + atomKey + "_lPass": create_deltaCore(
#                                         [str(headExpression[0]) + "_" + atomKey, str(headExpression) + "_" + atomKey])
#                                         for atomKey in leftatoms}
#                                     }
#         if type(headExpression[2]) == str:
#             rightskeletonCoresDict = {headExpression[2] + "_" + atomKey + "_r": create_deltaCore(
#                 colors=[headExpression[2] + "_" + atomKey, str(headExpression) + "_" + atomKey])
#                 for atomKey in candidatesDict[headExpression[2]]}
#             rightatoms = candidatesDict[headExpression[2]]
#         else:
#             rightskeletonCoresDict, rightatoms = skeleton_recursion(headExpression[2], candidatesDict)
#             rightskeletonCoresDict = {**rightskeletonCoresDict,
#                                      **{str(headExpression[2]) + "_" + atomKey + "_rPass": create_deltaCore(
#                                          [str(headExpression[2]) + "_" + atomKey, str(headExpression) + "_" + atomKey])
#                                          for atomKey in rightatoms}
#                                      }
#         return {**leftskeletonCoresDict, **rightskeletonCoresDict}, leftatoms + rightatoms
#
#
# def create_negationCoreDict(atoms, inprefix, outprefix):
#     negationMatrix = np.zeros(shape=(2, 2))
#     negationMatrix[0, 1] = 1
#     negationMatrix[1, 0] = 1
#
#     negationCoreDict = {}
#     for atomKey in atoms:
#         negationCoreDict[outprefix + "_" + atomKey + "_neg"] = cc.CoordinateCore(negationMatrix,
#                                                                                  [inprefix + atomKey,
#                                                                                   outprefix + atomKey],
#                                                                                  outprefix + atomKey + "_neg")
#     return negationCoreDict
#
#
# def create_deltaCore(colors, name=""):
#     values = np.zeros(shape=[2 for i in range(len(colors))])
#     values[tuple(0 for color in colors)] = 1
#     values[tuple(1 for color in colors)] = 1
#     return cc.CoordinateCore(values, colors, name)


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
