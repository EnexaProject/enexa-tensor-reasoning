import tnreason.logic.expression_calculus as ec
import tnreason.logic.coordinate_calculus as cc
import tnreason.logic.expression_generation as eg

import tnreason.representation.factdf_to_cores as ftoc
import tnreason.representation.sampledf_to_cores as stoc
import tnreason.representation.pairdf_to_cores as ptoc

import tnreason.optimization.generalized_als as gals
import tnreason.optimization.weight_estimation as wees

import tnreason.learning.expression_learning as el

import tnreason.model.create_mln as cmln

import numpy as np
import pgmpy


class MLNLearner:
    def __init__(self):
        self.weightedFormulas = []

    def learn_formula(self, skeletonExpression, candidatesDict, sampleDf, positiveCore, negativeCore):
        exLearner = el.AtomicLearner(skeletonExpression, candidatesDict)

        print(exLearner.candidatesDict)
        exLearner.generate_fixedCores_sampleDf(sampleDf)
        exLearner.generate_target_and_filterCore_from_exampleCores(positiveCore, negativeCore)
        exLearner.random_initialize_variableCoresDict()
        exLearner.als(10)
        exLearner.get_solution()

        solutionExpression = exLearner.solutionExpression
        atoms = np.unique(ec.get_variables(solutionExpression))
        atomDict = {
            atom : cc.CoordinateCore(stoc.sampleDf_to_universal_core(sampleDf,[atom]).flatten(),["j"]) for atom in atoms
        }
        solutionWeight = wees.calculate_weight(solutionExpression,atomDict)

        self.weightedFormulas.append([solutionExpression, solutionWeight])

    def generate_mln(self):
        expressionsDict = {str(i) : [self.weightedFormulas[i][0],self.weightedFormulas[i][1]] for i in range(len(self.weightedFormulas))}
        self.mln = cmln.create_markov_logic_network(expressionsDict)