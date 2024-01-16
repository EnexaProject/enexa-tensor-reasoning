from tnreason.model import formula_tensors as ft
from tnreason.model import model_visualization as mv

from tnreason.logic import expression_utils as eu
from tnreason.logic import coordinate_calculus as cc

from tnreason.contraction import core_contractor as coc

import numpy as np


class TensorRepresentation:
    def __init__(self, expressionsDict, headType="expFactor"):
        self.formulaTensorsDict = {formulaKey: ft.FormulaTensor(expression=expressionsDict[formulaKey][0],
                                                                formulaKey=formulaKey,
                                                                headType=headType,
                                                                weight=expressionsDict[formulaKey][1]
                                                                ) for formulaKey in expressionsDict}
        self.expressionsDict = expressionsDict
        self.headType = headType  ## To prevent resetting of headCores

        for formulaKey in expressionsDict:
            self.formulaTensorsDict[formulaKey].set_head(headType, weight=expressionsDict[formulaKey][1])

        self.atoms = np.unique(eu.get_all_variables([expressionsDict[formulaKey][0] for formulaKey in expressionsDict]))

    def add_expression(self, expression, weight=1, formulaKey=None):
        if formulaKey is None:
            formulaKey = str(expression)
        self.formulaTensorsDict[formulaKey] = ft.FormulaTensor(expression=expression, formulaKey=formulaKey,
                                                               headType=self.headType,
                                                               weight=weight)
        self.expressionsDict[formulaKey] = [expression, weight]
        self.atoms = np.unique(self.atoms, eu.get_variables(expression))

    def all_cores(self):
        allCoresDict = {}
        for formulaKey in self.formulaTensorsDict:
            allCoresDict = {**allCoresDict, **self.formulaTensorsDict[formulaKey].subExpressionCoresDict,
                            formulaKey + "_head": self.formulaTensorsDict[formulaKey].headCore}
        return allCoresDict

    def get_cores(self, formulaKeys=None, headType="expFactor"):
        if formulaKeys is None:
            formulaKeys = self.formulaTensorsDict.keys()
        if self.headType != headType:
            self.set_heads(headType)
        retCoresDict = {}
        for formulaKey in formulaKeys:
            retCoresDict = {**retCoresDict, **self.formulaTensorsDict[formulaKey].get_cores()}
        return retCoresDict

    def set_heads(self, headType):
        if self.headType != headType:
            print("Setting to {}".format(headType))
            for formulaKey in self.formulaTensorsDict.keys():
                self.formulaTensorsDict[formulaKey].set_head(headType, weight=self.expressionsDict[formulaKey][1])
            self.headType = headType

    def update_heads(self, updateDict, headType=None):
        if headType is None:
            headType = self.headType
        for formulaKey in updateDict:
            self.expressionsDict[formulaKey][1] = updateDict[formulaKey]
            self.formulaTensorsDict[formulaKey].set_head(headType, weight=updateDict[formulaKey])

    def get_weights(self):
        return {formulaKey: self.formulaTensorsDict[formulaKey].weight for formulaKey in self.formulaTensorsDict}

    def marginalized_contraction(self, atomList):
        marginalizationDict = {atomKey + "_marg": cc.CoordinateCore(np.ones(shape=(2)), [atomKey]) for atomKey in
                               self.atoms if atomKey not in atomList}
        margContractor = coc.CoreContractor({**self.all_cores(), **marginalizationDict}, openColors=atomList)
        return margContractor.contract()

    def contract_partition(self):
        if self.headType != "expFactor":
            print("Warning: Partition of Tensor Model computed, but headtype expFactor!")
        return self.marginalized_contraction([]).values

    def evidence_contraction(self, evidenceDict):
        inferedFormulaTensorDict = {formulaKey: self.formulaTensorsDict[formulaKey].infer_on_evidenceDict(evidenceDict)
                                    for formulaKey in self.formulaTensorsDict}
        resContractor = coc.CoreContractor(inferedFormulaTensorDict, openColors=[atomKey for atomKey in self.atoms if
                                                                                 atomKey not in evidenceDict])
        return resContractor.contract()

    def visualize(self, evidenceDict={}, strengthMultiplier=4, strengthCutoff=10, fontsize=10, showFormula=True,
                  pos=None):
        return mv.visualize_model(self.expressionsDict,
                                  strengthMultiplier=strengthMultiplier,
                                  strengthCutoff=strengthCutoff,
                                  fontsize=fontsize,
                                  showFormula=showFormula,
                                  evidenceDict=evidenceDict,
                                  pos=pos)


if __name__ == "__main__":
    learnedFormulaDict = {
        "f0": ["A1", 1],
        "f1": [["not", ["A2", "and", "A3"]], 1.2],
        "f2": ["A2", 2.1]
    }
    tRep = TensorRepresentation(learnedFormulaDict)

    tRep.visualize(evidenceDict={"A2": 1, "A3": False})
    print(tRep.evidence_contraction({"A2": 1, "A3": False}).values)
    print(tRep.marginalized_contraction(["A2", "A3"]).values)

    print(tRep.contract_partition())
