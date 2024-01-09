from tnreason.model import formula_tensors as ft

from tnreason.logic import expression_utils as eu
from tnreason.logic import coordinate_calculus as cc

from tnreason.contraction import core_contractor as coc

import numpy as np
class TensorRepresentation:
    def __init__(self, formulaDict, headType="expFactor"):
        self.formulaTensorsDict = {formulaKey : ft.FormulaTensor(formulaDict[formulaKey][0]) for formulaKey in formulaDict}
        for formulaKey in formulaDict:
            self.formulaTensorsDict[formulaKey].set_head(headType, weight=formulaDict[formulaKey][1])

        self.headsDict = {}
        self.atoms = np.unique(eu.get_all_variables([formulaDict[formulaKey][0] for formulaKey in formulaDict]))

    def all_cores(self):
        allCoresDict = {}
        for formulaKey in self.formulaTensorsDict:
            allCoresDict = {**allCoresDict, **self.formulaTensorsDict[formulaKey].subExpressionCoresDict, formulaKey+"_head": self.formulaTensorsDict[formulaKey].headCore}
        return allCoresDict

    def marginalized_contraction(self, atomList):
        marginalizationDict = {atomKey + "_marg": cc.CoordinateCore(np.ones(shape=(2)),[atomKey]) for atomKey in self.atoms if atomKey not in atomList}
        margContractor = coc.CoreContractor({**self.all_cores(), **marginalizationDict}, openColors=atomList)
        return margContractor.contract()

    def evidence_contraction(self, evidenceDict, headType="expFactor"):
        inferedFormulaTensorDict = {formulaKey : self.formulaTensorsDict[formulaKey].infer_on_evidenceDict(evidenceDict) for formulaKey in self.formulaTensorsDict}
        resContractor = coc.CoreContractor(inferedFormulaTensorDict, openColors=[atomKey for atomKey in self.atoms if atomKey not in evidenceDict])
        return resContractor.contract()



if __name__ == "__main__":
    learnedFormulaDict = {
        "f0": ["A1", 1],
        "f1": [["not", ["A2", "and", "A3"]], 1.2],
        "f2": ["A2", 2.1]
    }
    tRep = TensorRepresentation(learnedFormulaDict)
    print(tRep.evidence_contraction({"A2": 1, "A3": False}).values)

#    print(tRep.marginalize_on(["A2","A3"]).values)