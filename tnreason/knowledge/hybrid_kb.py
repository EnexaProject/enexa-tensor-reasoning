from tnreason.model import tensor_model as tm
from tnreason.model import logic_model as lm
from tnreason.model import formula_tensors as ft
from tnreason.model import create_cores as crc
from tnreason.model import sampling as samp
from tnreason.model import model_visualization as mov

from tnreason.contraction import core_contractor as coc

from tnreason.logic import expression_utils as eu

from tnreason.knowledge import storage

import numpy as np


def from_yaml(loadPath):
    modelSpec = storage.load_from_yaml(loadPath)
    return HybridKnowledgeBase(modelSpec["weightedFormulas"], factsDict=modelSpec["facts"])

class HybridKnowledgeBase:
    def __init__(self, weightedFormulasDict={}, factsDict={}):
        self.weightedFormulasDict = weightedFormulasDict.copy()
        self.factsDict = factsDict.copy()

        self.formulaTensors = tm.TensorRepresentation(weightedFormulasDict, headType="expFactor")
        self.facts = tm.TensorRepresentation(
            {key: [factsDict[key], None] for key in factsDict}, headType="truthEvaluation")
        self.atoms = list(
            eu.get_all_variables([weightedFormulasDict[key][0] for key in weightedFormulasDict] +
                                 [factsDict[key] for key in factsDict]))
        if not len(self.factsDict)==0:
            if not self.is_satisfiable():
                raise ValueError("The initialized Knowledge Base is inconsistent!")

    def is_satisfiable(self):
        return coc.CoreContractor(self.facts.get_cores(headType="truthEvaluation")).contract().values > 0

    def ask_constraint(self, constraint):
        probability = self.ask(constraint, evidenceDict={})
        if probability > 0.9999:
            return "entailed"
        elif probability == 0:
            return "contradicting"
        else:
            return "contingent"

    def tell_constraint(self, constraint, constraintKey=None):
        if constraintKey is None:
            constraintKey = "c" + str(len(self.factsDict))
        answer = self.ask_constraint(constraint)
        if answer == "entailed":
            print("{} is redundant to the Knowledge Base and has not been added.".format(constraint))
            return "not added"
        elif answer == "contradicting":
            print("{} would make the Knowledge Base inconsistent and has not been added.".format(constraint))
            return "not added"
        else:
            self.factsDict[constraintKey] = constraint
            self.facts.add_expression(constraint, None)

    def tell(self, formula, weight, formulaKey=None):
        if formulaKey is None:
            formulaKey = "f" + str(len(self.weightedFormulasDict))

        self.weightedFormulasDict[formulaKey] = [formula, weight]
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

    def exact_map_query(self, variableList, evidenceDict={}):
        distributionCore = self.query(variableList, evidenceDict)
        maxIndex = np.unravel_index(np.argmax(distributionCore.values.flatten()), distributionCore.values.shape)
        return {variable: maxIndex[i] for i, variable in enumerate(distributionCore.colors)}

    def annealed_map_query(self, variableList, evidenceDict={}, annealingPattern=[(10, 1)]):
        logRep = lm.LogicRepresentation(self.weightedFormulasDict, self.factsDict)
        logRep.infer(evidenceDict=evidenceDict, simplify=True)

        sampler = samp.GibbsSampler(*logRep.get_formulas_and_facts())
        return sampler.simulated_annealing_gibbs(variableList, annealingPattern)

    def to_yaml(self, savePath):
        storage.save_as_yaml({
            "weightedFormulas" : self.weightedFormulasDict,
            "facts" : self.factsDict
        }, savePath)

    def visualize(self, evidenceDict={}, strengthMultiplier=4, strengthCutoff=10, fontsize=10, showFormula=True,
                  pos=None):
        return mov.visualize_model(self.weightedFormulasDict,
                                   factsDict=self.factsDict,
                                  strengthMultiplier=strengthMultiplier,
                                  strengthCutoff=strengthCutoff,
                                  fontsize=fontsize,
                                  showFormula=showFormula,
                                  evidenceDict=evidenceDict,
                                  pos=pos)