from tnreason.model import logic_model as lm
from tnreason.model import model_visualization as mov

from tnreason.tensor import model_cores as crc, formula_tensors as ft, tensor_model as tm

from tnreason.logic import expression_utils as eu

from tnreason.encoding import storage

from tnreason.network import distributions as dist

import numpy as np

from tnreason import engine
from tnreason import encoding

defaultContractionMethod = "PgmpyVariableEliminator"


def from_yaml(loadPath):
    modelSpec = encoding.storage.load_from_yaml(loadPath)

    if "weightedFormulas" in modelSpec:
        weightedFormulas = modelSpec["weightedFormulas"]
    else:
        weightedFormulas = {}

    if "facts" in modelSpec:
        facts = modelSpec["facts"]
    else:
        facts = {}

    if "categoricalConstraints" in modelSpec:
        categoricalConstraints = modelSpec["categoricalConstraints"]
    else:
        categoricalConstraints = {}

    return HybridKnowledgeBase(weightedFormulasDict=weightedFormulas,
                               factsDict=facts,
                               categoricalConstraintsDict=categoricalConstraints)


class HybridKnowledgeBase:
    def __init__(self, weightedFormulasDict={}, factsDict={}, categoricalConstraintsDict={}):
        self.weightedFormulasDict = {key: [weightedFormulasDict[key][0], float(weightedFormulasDict[key][1])]
                                     for key in weightedFormulasDict}
        self.factsDict = factsDict.copy()
        self.categoricalConstraintsDict = categoricalConstraintsDict.copy()

        self.formulaTensors = tm.TensorRepresentation(weightedFormulasDict, headType="expFactor")
        self.facts = tm.TensorRepresentation(
            factsDict=factsDict,
            categoricalConstraintsDict=categoricalConstraintsDict,
            headType="truthEvaluation")
        self.atoms = list(
            eu.get_all_variables([weightedFormulasDict[key][0] for key in weightedFormulasDict] +
                                 [factsDict[key] for key in factsDict]))
        if not len(self.factsDict) == 0:
            if not self.is_satisfiable():
                raise ValueError("The initialized Knowledge Base is inconsistent!")

    def create_cores(self):
        structureCores = {**encoding.get_formulas_cores(
            {**{key: self.weightedFormulasDict[key][0] for key in self.weightedFormulasDict},
             **{key: self.factsDict[key] for key in self.factsDict}}
        ), **encoding.get_constraint_cores(self.categoricalConstraintsDict)}
        factHeadCores = {}
        for key in self.factsDict:
            factHeadCores = {**factHeadCores,
                             **encoding.get_head_core(expression=self.factsDict[key], headType="truthEvaluation")}

        probHeadCores = {}
        for key in self.weightedFormulasDict:
            probHeadCores = {**probHeadCores,
                             **encoding.get_head_core(expression=self.weightedFormulasDict[key][0],
                                                      headType="expFactor",
                                                      weight=self.weightedFormulasDict[key][1])}
        return {**structureCores, **factHeadCores, **probHeadCores}

    def include(self, secondHybridKB):
        ## Cannot handle key conflicts and does not include categoricalConstraints!
        for key in secondHybridKB.weightedFormulasDict:
            self.formulaTensors.add_expression(secondHybridKB.weightedFormulasDict[key][0],
                                               weight=float(secondHybridKB.weightedFormulasDict[key][1]),
                                               formulaKey=key)
        for key in secondHybridKB.factsDict:
            self.facts.add_expression(secondHybridKB.factsDict[key],
                                      weight=None,
                                      formulaKey=key)
        if not len(self.factsDict) == 0:
            if not self.is_satisfiable():
                raise ValueError("By including additional facts, the Knowledge Base got inconsistent!")

        self.weightedFormulasDict = {**self.weightedFormulasDict,
                                     **secondHybridKB.weightedFormulasDict}
        self.factsDict = {**self.factsDict,
                          **secondHybridKB.factsDict}
        self.atoms = list(set(self.atoms) | set(secondHybridKB.atoms))

    def is_satisfiable(self, contractionMethod=defaultContractionMethod):
        cores = encoding.get_formulas_cores(self.factsDict)
        for key in self.factsDict:
            cores = {**cores, **encoding.get_head_core(expression=self.factsDict[key], headType="truthEvaluation")}
        return engine.contract(method=contractionMethod, coreDict=cores, openColors=[]).values > 0

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

    def ask(self, queryFormula, evidenceDict={}, contractionMethod=defaultContractionMethod):
        overheadCount = len(
            [atom for atom in eu.get_variables(queryFormula) if (atom not in self.atoms and atom not in evidenceDict)])

        modelCores = {**self.formulaTensors.get_cores(),
                      **self.facts.get_cores(headType="truthEvaluation"),
                      **crc.create_evidenceCoresDict(evidenceDict)}
        return engine.contract(
            coreDict={**modelCores, **ft.FormulaTensor(queryFormula, headType="truthEvaluation").get_cores()},
            method=contractionMethod, openColors=[]).values / (
                    engine.contract(coreDict=modelCores, method=contractionMethod, openColors=[]).values * 2 ** overheadCount)

    def query(self, variableList, evidenceDict={}, contractionMethod=defaultContractionMethod):
        disconnectedVariables = [variable for variable in variableList if
                                 variable not in self.atoms and variable not in evidenceDict]

        return engine.contract(method=contractionMethod, coreDict=
            {
                **crc.create_emptyCoresDict(disconnectedVariables),
                **self.formulaTensors.get_cores(),
                **self.facts.get_cores(headType="truthEvaluation"),
                **crc.create_evidenceCoresDict(evidenceDict)
            },
            openColors=variableList).normalize()

    def exact_map_query(self, variableList, evidenceDict={}):
        distributionCore = self.query(variableList, evidenceDict)
        maxIndex = np.unravel_index(np.argmax(distributionCore.values.flatten()), distributionCore.values.shape)
        return {variable: maxIndex[i] for i, variable in enumerate(distributionCore.colors)}

    def annealed_map_query(self, variableList, evidenceDict={}, annealingPattern=[(10, 1), (5, 0.1), (2, 0.01)]):
        ## Need to support heating in distributions first!
        return self.gibbs_sample(variableList, evidenceDict)

    def gibbs_sample(self, variableList, evidenceDict={}, sweepNum=10):
        logRep = lm.LogicRepresentation(self.weightedFormulasDict, self.factsDict)
        logRep.infer(evidenceDict=evidenceDict, simplify=True)
        weightedFormulas, facts = logRep.get_formulas_and_facts()

        tenRep = tm.TensorRepresentation(expressionsDict=weightedFormulas,
                                         factsDict=facts,
                                         categoricalConstraintsDict=self.categoricalConstraintsDict)

        distribution = dist.TNDistribution(tenRep.get_cores())

        return distribution.gibbs_sampling(variableList, {variable: 2 for variable in variableList}, sweepNum=sweepNum)

    def evaluate_evidence(self, evidenceDict={}):
        return lm.LogicRepresentation(self.weightedFormulasDict, self.factsDict).evaluate_evidence(evidenceDict)

    def to_yaml(self, savePath):
        encoding.storage.save_as_yaml({
            "weightedFormulas": self.weightedFormulasDict,
            "facts": self.factsDict,
            "categoricalConstraints": self.categoricalConstraintsDict
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
