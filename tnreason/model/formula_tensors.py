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
        self.weight = weight

    def create_subExpressionCores(self):
        self.subExpressionCoresDict = crc.create_subExpressionCores(self.expression, str(self.formulaKey))

    def set_head(self, headType, weight=1):
        self.weight = weight
        self.headCore = crc.create_headCore(headType, weight, headColor=self.formulaKey + "_" + str(self.expression))

    def infer_on_evidenceDict(self, evidenceDict):
        return coc.CoreContractor({**self.get_cores(), **crc.create_evidenceCoresDict(evidenceDict)},
                                  openColors=[atomKey for atomKey in self.atoms if
                                              atomKey not in evidenceDict]).contract()

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
        crc.check_colorShapes([parameterCoresDict],
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
        self.selectorCoresDict = crc.create_selectorCoresDict(self.candidatesDict)

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
    def __init__(self, sampleDf, atoms=None, dataColor="j"):
        if atoms is None:
            self.atoms = sampleDf.columns
        else:
            self.atoms = atoms
        self.dataColor = dataColor

        self.create_dataCores(sampleDf)

    def create_dataCores(self, sampleDf):
        self.dataNum = sampleDf.values.shape[0]
        self.dataCores = {
            atomKey + "_data": crc.dataCore_from_sampleDf(sampleDf, atomKey, self.dataColor)
            for atomKey in self.atoms
        }

    def compute_shannon_entropy(self):
        contractedData = coc.CoreContractor(self.dataCores,
                                            openColors=self.atoms).contract().values.flatten() / self.dataNum
        logContractedData = np.log(np.copy(contractedData))
        logContractedData[logContractedData < -1e308] = 0
        return -np.dot(logContractedData, contractedData)

    def get_cores(self):
        return self.dataCores

class CategoricalConstraint:
    def __init__(self, atoms, name="categorical"):
        self.constraintCores = crc.create_constraintCoresDict(atoms, name)

    def get_cores(self):
        return self.constraintCores


if __name__ == "__main__":

    constraint = CategoricalConstraint(["a","b","c","d"])
    contraction = coc.CoreContractor(constraint.get_cores()).contract()
    print(contraction.values)
    exit()

    expression = ["A1", "and", ["not", "A2"]]
    fTensor = FormulaTensor(expression, "f1", headType="truthEvaluation", weight=1)

    contractor = coc.CoreContractor(fTensor.get_cores())
    contractor.optimize_coreList()
    contractor.create_instructionList_from_coreList()
    contractor.visualize()
