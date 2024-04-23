from tnreason import engine
from tnreason import encoding
from tnreason import algorithms

from tnreason.knowledge import knowledge_visualization as knv
from tnreason.knowledge import batch_evaluation as be

import pandas as pd

defaultContractionMethod = "PgmpyVariableEliminator"

probFormulasKey = "weightedFormulas"
logFormulasKey = "facts"
categoricalsKey = "categoricalConstraints"

entailedString = "entailed"
contradictingString = "contradicting"
contingentString = "contingent"


def load_kb_from_yaml(loadPath):
    modelSpec = encoding.load_from_yaml(loadPath)
    if probFormulasKey in modelSpec:
        weightedFormulas = modelSpec[probFormulasKey]
    else:
        weightedFormulas = {}

    if logFormulasKey in modelSpec:
        facts = modelSpec[logFormulasKey]
    else:
        facts = {}

    if categoricalsKey in modelSpec:
        categoricalConstraints = modelSpec[categoricalsKey]
    else:
        categoricalConstraints = {}

    return HybridKnowledgeBase(weightedFormulas=weightedFormulas,
                               facts=facts,
                               categoricalConstraints=categoricalConstraints)


class HybridKnowledgeBase:
    def __init__(self, weightedFormulas={}, facts={}, categoricalConstraints={}):
        self.weightedFormulasDict = {key: weightedFormulas[key][:-1] + [float(weightedFormulas[key][-1])]
                                     for key in weightedFormulas}
        self.factsDict = facts.copy()
        self.categoricalConstraintsDict = categoricalConstraints.copy()

        self.atoms = encoding.get_all_variables({**self.weightedFormulasDict, **self.factsDict})
        if not len(self.factsDict) == 0:
            if not self.is_satisfiable():
                raise ValueError("The initialized Knowledge Base is inconsistent!")

    def create_cores(self, evidenceDict={}, propagationReduction=False):
        if propagationReduction:
            propagator = be.KnowledgePropagator(self, evidenceDict=evidenceDict)
            propagator.evaluate()
            return propagator.find_carrying_cores()
        else:
            return {**encoding.create_formulas_cores({**self.weightedFormulasDict, **self.factsDict}),
                    **encoding.create_constraints(self.categoricalConstraintsDict),
                    **encoding.create_evidence_cores(evidenceDict)}

    def partitionFunction(self, contractionMethod=defaultContractionMethod):
        return engine.contract(method=contractionMethod, coreDict=self.create_cores(), openColors=[]).values

    def include(self, secondHybridKB):

        if not len(self.factsDict) == 0:
            if not self.is_satisfiable():
                raise ValueError("By including additional facts, the Knowledge Base got inconsistent!")

        self.weightedFormulasDict = {**self.weightedFormulasDict,
                                     **secondHybridKB.weightedFormulasDict}
        self.factsDict = {**self.factsDict,
                          **secondHybridKB.factsDict}
        self.atoms = list(set(self.atoms) | set(secondHybridKB.atoms))

    def is_satisfiable(self, contractionMethod=defaultContractionMethod):
        return engine.contract(method=contractionMethod, coreDict={**encoding.create_formulas_cores(self.factsDict),
                                                                   **encoding.create_constraints(
                                                                       self.categoricalConstraintsDict)},
                               openColors=[]).values > 0

    def ask_constraint(self, constraint):
        probability = self.ask(constraint, evidenceDict={})
        if probability > 0.9999:
            return entailedString
        elif probability == 0:
            return contradictingString
        else:
            return contingentString

    def tell_constraint(self, constraint, constraintKey=None):
        if constraintKey is None:
            constraintKey = "c" + str(len(self.factsDict))
        answer = self.ask_constraint(constraint)
        if answer == entailedString:
            print("{} is redundant to the Knowledge Base and has not been added.".format(constraint))
            return entailedString
        elif answer == contradictingString:
            print("{} would make the Knowledge Base inconsistent and has not been added.".format(constraint))
            return contradictingString
        else:
            self.factsDict[constraintKey] = constraint
            return contingentString

    def tell(self, formula, weight, formulaKey=None):
        if formulaKey is None:
            formulaKey = "f" + str(len(self.weightedFormulasDict))

        self.weightedFormulasDict[formulaKey] = [formula, weight]

        for atom in encoding.get_variables(formula):
            if atom not in self.atoms:
                self.atoms.append(atom)

    def ask(self, queryFormula, evidenceDict={}, contractionMethod=defaultContractionMethod):

        contracted = engine.contract(
            coreDict={**encoding.create_formulas_cores({**self.weightedFormulasDict, **self.factsDict}),
                      **encoding.create_evidence_cores(evidenceDict),
                      **encoding.create_constraints(self.categoricalConstraintsDict),
                      **encoding.create_raw_formula_cores(queryFormula)
                      },
            method=contractionMethod, openColors=[encoding.get_formula_color(queryFormula)]).values

        return contracted[1] / (contracted[0] + contracted[1])

    def query(self, variableList, evidenceDict={}, contractionMethod=defaultContractionMethod):
        return engine.contract(method=contractionMethod, coreDict={
            **encoding.create_emptyCoresDict([variable for variable in variableList if
                                              variable not in self.atoms and variable not in evidenceDict]),
            **encoding.create_formulas_cores({**self.weightedFormulasDict, **self.factsDict}),
            **encoding.create_evidence_cores(evidenceDict),
            **encoding.create_constraints(self.categoricalConstraintsDict)
        }, openColors=variableList).normalize()

    def exact_map_query(self, variableList, evidenceDict={}):
        distributionCore = self.query(variableList, evidenceDict)
        maxIndex = distributionCore.get_maximal_index()
        return {variable: maxIndex[i] for i, variable in enumerate(distributionCore.colors)}

    def annealed_sample(self, variableList, evidenceDict={}, annealingPattern=[[10, 1]]):
        weightedFormulas, facts = self.weightedFormulasDict, self.factsDict

        sampler = algorithms.Gibbs({**encoding.create_formulas_cores({**weightedFormulas, **facts}),
                                    **encoding.create_constraints(self.categoricalConstraintsDict),
                                    **encoding.create_evidence_cores(evidenceDict)})

        sampler.ones_initialization(updateKeys=variableList, shapesDict={variable: 2 for variable in variableList},
                                    colorsDict={variable: [variable] for variable in variableList})

        return sampler.annealed_sample(updateKeys=variableList, annealingPattern=annealingPattern)

    def create_sampleDf(self, sampleNum, variableList=None, annealingPattern=[[10, 1]], outType="int64"):
        if variableList is None:
            variableList = self.atoms
        sampleDf = pd.DataFrame(columns=variableList)
        for samplePos in range(sampleNum):
            sampleDf = pd.concat(
                [sampleDf,
                 pd.DataFrame(self.annealed_sample(variableList=variableList, annealingPattern=annealingPattern),
                              index=[samplePos])])
        return sampleDf.astype(outType)

    def to_yaml(self, savePath):
        encoding.storage.save_as_yaml({
            probFormulasKey: self.weightedFormulasDict,
            logFormulasKey: self.factsDict,
            categoricalsKey: self.categoricalConstraintsDict
        }, savePath)

    def evaluate_evidence(self, evidenceDict):
        propagator = be.KnowledgePropagator(self, evidenceDict=evidenceDict)
        return propagator.evaluate()

    def visualize(self, evidenceDict={}, strengthMultiplier=4, strengthCutoff=10, fontsize=10, showFormula=True,
                  pos=None):
        return knv.visualize_knowledge(expressionsDict=self.weightedFormulasDict,
                                       factsDict=self.factsDict,
                                       strengthMultiplier=strengthMultiplier,
                                       strengthCutoff=strengthCutoff,
                                       fontsize=fontsize,
                                       showFormula=showFormula,
                                       evidenceDict=evidenceDict,
                                       pos=pos)
