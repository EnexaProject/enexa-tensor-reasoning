from tnreason.logic import expression_utils as eu

from tnreason import contraction
from tnreason.contraction import core_contractor as coc
from tnreason.tensor import model_cores as crc

import numpy as np


class FormulaTensor:
    '''
    expression:
    formulaKey: For distinction , typically passed by TensorRepresentation as dict key
    headType: Whether \weight\ftensor (truthEvaluation, default) or \expof{\weight\ftensor} (expFactor) or \partial\expof{\weight\ftensor} (diffExpFactor) is created
    weight: Factor on the formulaTensor
    '''

    def __init__(self, expression, formulaKey=None, headType="weightedTruthEvaluation", weight=1,
                 coreType="NumpyTensorCore"):
        if formulaKey is not None:
            self.formulaKey = formulaKey
        else:
            self.formulaKey = str(expression)
        self.expression = expression
        self.atoms = eu.get_variables(expression)

        ## Build the Cores
        self.create_conCores(coreType)
        self.weight = weight
        self.set_head(headType=headType, weight=weight)

    def create_conCores(self, coreType):
        self.conCores = crc.create_conCores(self.expression, coreType=coreType)

    def set_head(self, headType, weight=None):
        if weight is None:
            weight = self.weight
        else:
            self.weight = weight
        self.headType = headType
        self.headCore = crc.create_headCore(self.expression, headType, weight)

    def infer_on_evidenceDict(self, evidenceDict, contractionMethod="PgmpyVariableEliminator"):
        return contraction.get_contractor(contractionMethod)(
            {**self.get_cores(), **crc.create_evidenceCoresDict(evidenceDict)},
            openColors=[atomKey for atomKey in self.atoms if
                        atomKey not in evidenceDict]).contract()

    def get_cores(self, headType=None):
        if headType is None:
            headType = self.headType
        if headType != self.headType:
            self.set_head(headType)
        return {**self.conCores,
                **self.headCore}


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

    def compute_shannon_entropy(self, contractionMethod="PgmpyVariableEliminator"):
        contractedData = contraction.get_contractor(contractionMethod)(
            {**self.dataCores.copy(),
             **change_color_in_coredict(self.dataCores, {self.dataColor: self.dataColor + "_out"})},
            openColors=[self.dataColor + "_out"]
        ).contract().values

        assert not np.isnan(contractedData).any(), "Contraction for Entropy did not work!"

        logContractedData = np.log(contractedData / self.dataNum)
        return -np.sum(logContractedData) / self.dataNum

    def get_cores(self):
        return self.dataCores


class CategoricalConstraint:
    def __init__(self, atoms, name="categorical"):
        self.constraintCores = crc.create_constraintCoresDict(atoms, name)

    def get_cores(self):
        return self.constraintCores


def change_color_in_coredict(coreDict, colorReplaceDict, replaceSuffix="_replaced"):
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


if __name__ == "__main__":
    constraint = CategoricalConstraint(["a", "b", "c", "d"])
    contraction = coc.CoreContractor(constraint.get_cores()).contract()
    print(contraction.values)
    exit()

    expression = ["A1", "and", ["not", "A2"]]
    fTensor = FormulaTensor(expression, "f1", headType="truthEvaluation", weight=1)

    contractor = coc.CoreContractor(fTensor.get_cores())
    contractor.optimize_coreList()
    contractor.create_instructionList_from_coreList()
    contractor.visualize()
