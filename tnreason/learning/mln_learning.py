import tnreason.logic.expression_calculus as ec
import tnreason.logic.coordinate_calculus as cc
import tnreason.logic.expression_generation as eg

import tnreason.representation.factdf_to_cores as ftoc
import tnreason.representation.sampledf_to_cores as stoc
import tnreason.representation.pairdf_to_cores as ptoc

import tnreason.optimization.weight_estimation as wees

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
            atom: cc.CoordinateCore(stoc.sampleDf_to_universal_core(sampleDf, [atom]).flatten(), ["j"]) for atom in
            atoms
        }

    def learn_independent_weight(self, expression):
        return wees.calculate_weight(expression, self.atomDict)

    def add_independent_formula(self, expression):
        weight = self.learn_independent_weight(expression)
        self.weightedFormulas.append([expression, weight])
        print("The expression {} with weight {} has been added.".format(expression, weight))

    def learn_implication(self, positiveExpression, skeletonExpression, candidatesDict):
        positiveCore = ec.evaluate_expression_on_sampleDf(self.sampleDf, positiveExpression)
        negativeCore = positiveCore.negate()

        learnedPremise = self.learn_formula(skeletonExpression, candidatesDict, positiveCore, negativeCore)
        solutionExpression = eg.generate_from_generic_expression([learnedPremise, "imp", positiveExpression])

        self.add_independent_formula(solutionExpression)

    def learn_equivalence(self, positiveExpression, skeletonExpression, candidatesDict):
        positiveCore = ec.evaluate_expression_on_sampleDf(self.sampleDf, positiveExpression)
        negativeCore = positiveCore.negate()

        learnedPremise = self.learn_formula(skeletonExpression, candidatesDict, positiveCore, negativeCore)
        solutionExpression = eg.generate_from_generic_expression([learnedPremise, "eq", positiveExpression])

        self.add_independent_formula(solutionExpression)

    def learn_tautology(self, skeletonExpression, candidatesDict):
        sampleNum = self.sampleDf.values.shape[0]
        positiveCore = cc.CoordinateCore(np.ones(sampleNum),["j"])
        negativeCore = positiveCore.negate()

        solutionExpression = self.learn_formula(skeletonExpression, candidatesDict, positiveCore, negativeCore)

        self.add_independent_formula(solutionExpression)

    def learn_formula(self, skeletonExpression, candidatesDict, positiveCore, negativeCore):
        exLearner = el.AtomicLearner(skeletonExpression, candidatesDict)

        exLearner.generate_fixedCores_sampleDf(self.sampleDf)
        exLearner.generate_target_and_filterCore_from_exampleCores(positiveCore, negativeCore)
        exLearner.random_initialize_variableCoresDict()
        exLearner.als(10)
        exLearner.get_solution()

        return exLearner.solutionExpression

    def generate_mln(self):
        return mln.MarkovLogicNetwork(expressionsDict=
                                      {str(i): [self.weightedFormulas[i][0], self.weightedFormulas[i][1]]
                                       for i in range(len(self.weightedFormulas))})