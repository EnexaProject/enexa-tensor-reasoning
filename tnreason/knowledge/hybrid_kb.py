from tnreason.model import tensor_model as tm
from tnreason.model import formula_tensors as ft
from tnreason.model import create_cores as crc

from tnreason.contraction import core_contractor as coc

from tnreason.logic import expression_utils as eu

import numpy as np


## To be implemented:
# Combine hardFormulas
class HybridKnowledgeBase:
    def __init__(self, weightedFormulasDict={}, factsList=[]):
        self.formulaTensors = tm.TensorRepresentation(weightedFormulasDict, headType="expFactor")
        self.facts = tm.TensorRepresentation(
            {"c" + str(i): [constraint, None] for i, constraint in enumerate(factsList)},
            headType="truthEvaluation")
        self.atoms = list(
            eu.get_all_variables([weightedFormulasDict[key][0] for key in weightedFormulasDict] + factsList))
        if not self.is_satisfiable():
            raise ValueError("The initialized Knowledge Base is inconsistent!")

    def is_satisfiable(self):
        return coc.CoreContractor(self.facts.get_cores(headType="truthEvaluation")).contract().values > 0

    def ask_constraint(self, constraint):
        probability = self.ask(constraint, evidenceDict={})
        if probability == 1:
            return "entailed"
        elif probability == 0:
            return "contradicting"
        else:
            return "contingent"

    def tell_constraint(self, constraint):
        answer = self.ask_constraint(constraint)
        if answer == "entailed":
            print("{} is redundant to the Knowledge Base and has not been added.".format(constraint))
            return "not added"
        elif answer == "contradicting":
            print("{} would make the Knowledge Base inconsistent and has not been added.".format(constraint))
            return "not added"
        else:
            self.facts.add_expression(constraint, None)

    def tell(self, formula, weight, formulaKey=None):
        if formulaKey is None:
            formulaKey = "f" + str(len(self.formulaTensors.expressionsDict))
        self.formulaTensors.add_expression(formula, weight, formulaKey)

        for atom in eu.get_variables(formula):
            if atom not in self.atoms:
                self.atoms.append(atom)

    def ask(self, queryFormula, evidenceDict={}):
        overheadCount = len(
            [atom for atom in eu.get_variables(queryFormula) if (atom not in self.atoms and atom not in evidenceDict)])

        modelCores = {**self.formulaTensors.get_cores(),
                      **self.facts.get_cores(headType="truthEvaluation"),
                      **crc.create_evidenceCoresDict(evidenceDict)}

        return coc.CoreContractor(
            {**modelCores,
             **ft.FormulaTensor(queryFormula,
                                headType="truthEvaluation").get_cores()}).contract().values / (
                coc.CoreContractor(
                    modelCores).contract().values * 2 ** overheadCount)

    def query(self, variableList, evidenceDict={}):
        return coc.CoreContractor(
            {
                **self.formulaTensors.get_cores(),
                **self.facts.get_cores(headType="truthEvaluation"),
                **crc.create_evidenceCoresDict(evidenceDict)
            },
            openColors=variableList).contract().normalize()

    def map_query(self, variableList, evidenceDict={}):
        distributionCore = self.query(variableList, evidenceDict)
        maxIndex = np.unravel_index(np.argmax(distributionCore.values.flatten()), distributionCore.values.shape)
        return {variable: maxIndex[i] for i, variable in enumerate(distributionCore.colors)}
