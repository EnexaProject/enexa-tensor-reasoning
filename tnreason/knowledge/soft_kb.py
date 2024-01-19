from tnreason.model import tensor_model as tm
from tnreason.model import formula_tensors as ft
from tnreason.model import create_cores as crc

from tnreason.contraction import core_contractor as coc

from tnreason.logic import expression_utils as eu


class SoftKnowledgeBase:
    def __init__(self, weightedFormulasDict):
        self.weightedFormulasDict = weightedFormulasDict
        self.formulaTensors = tm.TensorRepresentation(weightedFormulasDict, headType="expFactor")
        self.atoms = list(eu.get_all_variables([weightedFormulasDict[key][0] for key in weightedFormulasDict]))

    ## Appends a weighted formula to the Knowledge Base
    def tell(self, formula, weight, formulaKey=None):
        if formulaKey is None:
            formulaKey = "f" + str(len(self.weightedFormulasDict))
        self.weightedFormulasDict[formulaKey] = [formula, weight]
        self.formulaTensors.add_expression(formula, weight, formulaKey)

        for atom in eu.get_variables(formula):
            if atom not in self.atoms:
                self.atoms.append(atom)

    ## Gives the probability of formula having formula satisfied given the evidence
    # quotient of model contraction with formula and without
    # overheadCount compensates atoms only in queryFormula
    def ask(self, queryFormula, evidenceDict={}):
        overheadCount = len(
            [atom for atom in eu.get_variables(queryFormula) if (atom not in self.atoms and atom not in evidenceDict)])

        modelCores = {**self.formulaTensors.get_cores(),
                      **crc.create_evidenceCoresDict(evidenceDict)}

        return coc.CoreContractor(
            {**modelCores, **ft.FormulaTensor(queryFormula,
                                              headType="truthEvaluation").get_cores()}).contract().values / (
                coc.CoreContractor(
                    modelCores).contract().values * 2 ** overheadCount)
