import tnreason.logic.expression_calculus as ec
import tnreason.logic.coordinate_calculus as cc
import tnreason.logic.expression_generation as eg

import tnreason.representation.factdf_to_cores as ftoc
import tnreason.representation.sampledf_to_cores as stoc
import tnreason.representation.pairdf_to_cores as ptoc

import tnreason.optimization.weight_estimation as wees
import tnreason.optimization.expression_refinement as er

import tnreason.learning.expression_learning as el

from tnreason.model import markov_logic_network as mln

import numpy as np


class AtomicMLNLearner:
    def __init__(self):
        self.weightedFormulas = []
        self.sampleDf = None
        self.atomDict = None

    def load_sampleDf(self, sampleDf):
        self.sampleDf = sampleDf

        atoms = list(sampleDf.columns)
        self.atomDict = {
            atom: cc.CoordinateCore(stoc.sampleDf_to_universal_core(sampleDf, [atom]).flatten(), ["j"])
            for atom in atoms
        }

    def learn_independent_weight(self, expression, verbose=False):
        return wees.calculate_weight(expression, self.atomDict, verbose=verbose)

    def alternating_weight_optimization(self, sweepNum):
        estimator = wees.WeightEstimator([weightedFormula[0] for weightedFormula in self.weightedFormulas])
        estimator.alternating_optimization(self.atomDict, sweepNum)
        self.weightedFormulas = [[estimator.formulaDict[formulaKey][0], estimator.formulaDict[formulaKey][3]] for
                                 formulaKey in estimator.formulaDict]

    def add_independent_formula(self, expression, criterion=""):
        empRate, satRate, weight = self.learn_independent_weight(expression)
        if criterion_satisfied(empRate, satRate, weight, criterion):
            self.weightedFormulas.append([expression, weight])
            print("The expression {} with weight {} has been added.".format(expression, weight))
        else:
            print(
                "Expression {} with weight {} does not satisfy the criterion {}.".format(expression, weight, criterion))

    def learn_implication(self, positiveExpression, skeletonExpression, candidatesDict,
                          acceptanceCriterion="weight>0.5", refinement_left=0):
        positiveCore = ec.evaluate_expression_on_sampleDf(self.sampleDf, positiveExpression)
        negativeCore = positiveCore.negate()

        learnedPremise = self.learn_formula_with_refinement(skeletonExpression, candidatesDict, positiveCore,
                                                            negativeCore, refinement_left, acceptanceCriterion)
        solutionExpression = eg.generate_from_generic_expression([learnedPremise, "imp", positiveExpression])

        self.add_independent_formula(solutionExpression, criterion=acceptanceCriterion)

    def learn_equivalence(self, positiveExpression, skeletonExpression, candidatesDict,
                          acceptanceCriterion="weight>0.5", refinement_left=0):
        positiveCore = ec.evaluate_expression_on_sampleDf(self.sampleDf, positiveExpression)
        negativeCore = positiveCore.negate()

        learnedPremise = self.learn_formula_with_refinement(skeletonExpression, candidatesDict, positiveCore,
                                                            negativeCore, refinement_left, acceptanceCriterion)
        solutionExpression = eg.generate_from_generic_expression([learnedPremise, "eq", positiveExpression])

        self.add_independent_formula(solutionExpression, criterion=acceptanceCriterion)

    def learn_tautology(self, skeletonExpression, candidatesDict, acceptanceCriterion="weight>0.5", refinement_left=0):
        sampleNum = self.sampleDf.values.shape[0]
        positiveCore = cc.CoordinateCore(np.ones(sampleNum), ["j"])
        negativeCore = positiveCore.negate()

        solutionExpression = self.learn_formula_with_refinement(skeletonExpression, candidatesDict, positiveCore,
                                                                negativeCore, refinement_left, acceptanceCriterion)

        self.add_independent_formula(solutionExpression, criterion=acceptanceCriterion)

    def learn_formula(self, skeletonExpression, candidatesDict, positiveCore, negativeCore, balance=True):
        exLearner = el.AtomicLearner(skeletonExpression, candidatesDict)

        exLearner.generate_fixedCores_sampleDf(self.sampleDf)
        exLearner.generate_target_and_filterCore_from_exampleCores(positiveCore, negativeCore)
        if balance:
            exLearner.balance_importance(positiveCore=positiveCore, negativeCore=negativeCore)
        exLearner.random_initialize_variableCoresDict()
        exLearner.als(10)
        exLearner.get_solution()

        return exLearner.solutionExpression

    def learn_formula_with_refinement(self, skeletonExpression, candidatesDict, positiveCore, negativeCore,
                                      refinement_left=0, acceptance_criterion="weight>0.5"):
        solutionExpression = self.learn_formula(skeletonExpression, candidatesDict, positiveCore, negativeCore)

        empRate, satRate, weight = self.learn_independent_weight(solutionExpression)
        if refinement_left > 0 and not criterion_satisfied(empRate, satRate, weight, acceptance_criterion):
            print("### {} Refinements left ###".format(refinement_left))
            newSkeleton = er.add_leaf_atom(skeletonExpression)
            refinement_left -= 1
            return self.learn_formula_with_refinement(newSkeleton, candidatesDict, positiveCore, negativeCore,
                                                      refinement_left)
        else:
            return solutionExpression

    def generate_mln(self):
        return mln.MarkovLogicNetwork(expressionsDict=
                                      {str(i): [self.weightedFormulas[i][0], self.weightedFormulas[i][1]]
                                       for i in range(len(self.weightedFormulas))})


def criterion_satisfied(empRate, satRate, weight, criterion):
    criteria = criterion.split(",")
    for criterion in criteria:
        if criterion.startswith("weight"):
            threshold = float(criterion.split(">")[1])
            if weight < threshold:
                print("Acceptance Criterion {} failed.".format(criterion))
                return False
        elif criterion.startswith("empRate"):
            threshold = float(criterion.split(">")[1])
            if empRate < threshold:
                print("Acceptance Criterion {} failed.".format(criterion))
                return False
        elif criterion.startswith("satRate"):
            threshold = float(criterion.split("<")[1])
            if satRate > threshold:
                print("Acceptance Criterion {} failed.".format(criterion))
                return False
        else:
            raise ValueError("Acceptance Criterion {} not understood.".format(criterion))
    return True
