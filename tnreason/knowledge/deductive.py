from tnreason import engine
from tnreason import encoding
from tnreason import algorithms

from tnreason.knowledge import knowledge_visualization as knv
from tnreason.knowledge import batch_evaluation as be

import pandas as pd

defaultContractionMethod = "PgmpyVariableEliminator"

entailedString = "entailed"
contradictingString = "contradicting"
contingentString = "contingent"


class HybridInferer:
    def __init__(self, hybridKB):
        self.hybridKB = hybridKB

    def create_cores(self, evidenceDict={}, propagationReduction=False):
        if propagationReduction:
            propagator = be.KnowledgePropagator(self.hybridKB, evidenceDict=evidenceDict)
            propagator.evaluate()
            return propagator.find_carrying_cores()
        else:
            return self.hybridKB.create_cores()

    def partitionFunction(self, contractionMethod=defaultContractionMethod):
        return engine.contract(method=contractionMethod, coreDict=self.create_cores(), openColors=[]).values

    def is_satisfiable(self, contractionMethod=defaultContractionMethod):
        return engine.contract(method=contractionMethod, coreDict=self.hybridKB.create_cores(hardOnly=True),
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
            constraintKey = "c" + str(len(self.hybridKB.facts))
        answer = self.ask_constraint(constraint)
        if answer == entailedString:
            print("{} is redundant to the Knowledge Base and has not been added.".format(constraint))
            return entailedString
        elif answer == contradictingString:
            print("{} would make the Knowledge Base inconsistent and has not been added.".format(constraint))
            return contradictingString
        else:
            self.hybridKB.facts[constraintKey] = constraint
            return contingentString

    def tell(self, formula, weight, formulaKey=None):
        if formulaKey is None:
            formulaKey = "f" + str(len(self.hybridKB.weightedFormulas))

        self.hybridKB.weightedFormulas[formulaKey] = [formula, weight]

        for atom in encoding.get_variables(formula):
            if atom not in self.hybridKB.atoms:
                self.hybridKB.atoms.append(atom)

    def ask(self, queryFormula, evidenceDict={}, contractionMethod=defaultContractionMethod):

        contracted = engine.contract(
            coreDict={**encoding.create_formulas_cores({**self.hybridKB.weightedFormulas, **self.hybridKB.facts}),
                      **encoding.create_evidence_cores(evidenceDict),
                      **encoding.create_constraints(self.hybridKB.categoricalConstraints),
                      **encoding.create_raw_formula_cores(queryFormula)
                      },
            method=contractionMethod, openColors=[encoding.get_formula_color(queryFormula)]).values

        return contracted[1] / (contracted[0] + contracted[1])

    def query(self, variableList, evidenceDict={}, contractionMethod=defaultContractionMethod):
        return engine.contract(method=contractionMethod, coreDict={
            **encoding.create_emptyCoresDict([variable for variable in variableList if
                                              variable not in self.hybridKB.atoms and variable not in evidenceDict]),
            **encoding.create_formulas_cores({**self.hybridKB.weightedFormulas, **self.hybridKB.facts}),
            **encoding.create_evidence_cores(evidenceDict),
            **encoding.create_constraints(self.hybridKB.categoricalConstraints)
        }, openColors=variableList).normalize()

    def exact_map_query(self, variableList, evidenceDict={}):
        distributionCore = self.query(variableList, evidenceDict)
        maxIndex = distributionCore.get_maximal_index()
        return {variable: maxIndex[i] for i, variable in enumerate(distributionCore.colors)}

    def annealed_sample(self, variableList, evidenceDict={}, annealingPattern=[[10, 1]]):
        weightedFormulas, facts = self.hybridKB.weightedFormulas, self.hybridKB.facts

        sampler = algorithms.Gibbs({**encoding.create_formulas_cores({**weightedFormulas, **facts}),
                                    **encoding.create_constraints(self.hybridKB.categoricalConstraints),
                                    **encoding.create_evidence_cores(evidenceDict)})

        sampler.ones_initialization(updateKeys=variableList, shapesDict={variable: 2 for variable in variableList},
                                    colorsDict={variable: [variable] for variable in variableList})

        return sampler.annealed_sample(updateKeys=variableList, annealingPattern=annealingPattern)

    def create_sampleDf(self, sampleNum, variableList=None, annealingPattern=[[10, 1]], outType="int64"):
        if variableList is None:
            variableList = self.hybridKB.atoms
        sampleDf = pd.DataFrame(columns=variableList)
        for samplePos in range(sampleNum):
            sampleDf = pd.concat(
                [sampleDf,
                 pd.DataFrame(self.annealed_sample(variableList=variableList, annealingPattern=annealingPattern),
                              index=[samplePos])])
        return sampleDf.astype(outType)

    def evaluate_evidence(self, evidenceDict):
        propagator = be.KnowledgePropagator(self.hybridKB, evidenceDict=evidenceDict)
        return propagator.evaluate()

    def visualize(self, evidenceDict={}):
        return knv.visualize_knowledge(expressionsDict=self.hybridKB.weightedFormulas,
                                       factsDict=self.hybridKB.facts,
                                       evidenceDict=evidenceDict)
