import tnreason.logic.expression_calculus as ec
import tnreason.logic.coordinate_calculus as cc
import tnreason.logic.expression_generation as eg

import tnreason.representation.factdf_to_cores as ftoc
import tnreason.representation.sampledf_to_cores as stoc
import tnreason.representation.pairdf_to_cores as ptoc

import tnreason.optimization.generalized_als as gals

from tnreason.model import markov_logic_network

import numpy as np


class LearnerBase:
    def __init__(self, skeletonExpression, candidatesDict=None):
        self.skeleton = skeletonExpression
        self.skeletonAtoms = ec.get_variables(skeletonExpression)
        self.candidatesDict = candidatesDict

        self.targetCore = None
        self.filterCore = None

    def generate_target_and_filterCore_from_exampleCores(self, positiveCore, negativeCore=None):
        if negativeCore is None:
            negativeCore = positiveCore.create_constant([], zero=True)
        self.targetCore = positiveCore.clone()
        self.filterCore = positiveCore.compute_or(negativeCore)

    def als(self, sweepnum):
        self.optimizer = gals.GeneralizedALS(self.variablesCoresDict, self.fixedCoresDict)
        self.optimizer.set_targetCore(self.targetCore)
        self.optimizer.set_filterCore(self.filterCore)

        self.optimizer.sweep(sweepnum=sweepnum, contractionScheme=self.skeleton)
        self.variablesCoresDict = self.optimizer.variableCoresDict

    def get_solution(self):
        self.solutionDict = {}
        for legKey in self.variablesCoresDict:
            maxPos = np.argmax([abs(val) for val in self.variablesCoresDict[legKey].values])
            self.solutionDict[legKey] = self.candidatesDict[legKey][maxPos]
        self.solutionExpression = eg.replace_atoms(self.skeleton, self.solutionDict)

    @property
    def factor_core(self):
        return markov_logic_network.calculate_dangling_basis(self.solutionExpression).calculate_truth()


class AtomicLearner(LearnerBase):

    def generate_fixedCores_sampleDf(self, sampleDf, candidatesDict=None):
        if candidatesDict is not None:
            self.candidatesDict = candidatesDict

        self.fixedCoresDict = {}

        for atomKey in self.skeletonAtoms:
            values = stoc.sampleDf_to_universal_core(sampleDf, self.candidatesDict[atomKey])
            self.fixedCoresDict[atomKey] = cc.CoordinateCore(values, ["j", atomKey])

    def random_initialize_variableCoresDict(self):
        self.variablesCoresDict = {}
        for legKey in self.fixedCoresDict:
            varColors, varDims = atomic_find_var(self.fixedCoresDict[legKey])
            self.variablesCoresDict[legKey] = cc.CoordinateCore(np.random.random(varDims), varColors)

    def set_targetCore(self, targetCore=None, length=0, targetIsFilter=True):
        if targetCore is None:
            self.targetCore = cc.CoordinateCore(np.ones([length]), ["j"])
        else:
            self.targetCore = targetCore

        if targetIsFilter:
            self.filterCore = self.targetCore.clone()


class VariableLearner(LearnerBase):

    def generate_fixedCores_sampledf(self, sampleDf):
        self.candidatesDict = {}
        self.fixedCoresDict = {}

        for atomKey in self.skeletonAtoms:
            if "," in atomKey:
                indSpec = atomKey.split("(")[1][:-1]
                indKey1, indKey2 = indSpec.split(",")
                relationKey = atomKey.split("(")[0]

                coreValues, candidates, latency = stoc.sampleDf_to_relation_values(sampleDf, indKey1, indKey2)

                self.candidatesDict[atomKey] = candidates
                self.fixedCoresDict[atomKey] = cc.CoordinateCore(coreValues, [indKey1, relationKey, indKey2])
            else:
                indKey = atomKey.split("(")[1][:-1]
                classKey = atomKey.split("(")[0]

                coreValues, candidates, latency = stoc.sampleDf_to_class_values(sampleDf, indKey)

                self.candidatesDict[atomKey] = candidates
                self.fixedCoresDict[atomKey] = cc.CoordinateCore(coreValues, [indKey, classKey])

    def generate_fixedCores_factDf(self, df, individualsDict, candidatesDict, prefix=""):
        self.candidatesDict = candidatesDict
        self.fixedCoresDict = {}

        for atomKey in self.skeletonAtoms:
            if "," in atomKey:
                indSpec = atomKey.split("(")[1][:-1]
                indKey1, indKey2 = indSpec.split(",")
                relationKey = atomKey.split("(")[0]
                if not relationKey.startswith("R"):
                    relationKey = "R_" + relationKey
                coreValues, latency = ftoc.factDf_to_relation_values(df,
                                                                     individuals1=individualsDict[indKey1],
                                                                     individuals2=individualsDict[indKey2],
                                                                     relations=candidatesDict[atomKey],
                                                                     prefix=prefix)
                self.fixedCoresDict[atomKey] = cc.CoordinateCore(coreValues, [indKey1, relationKey, indKey2])

            else:
                indKey = atomKey.split("(")[1][:-1]
                classKey = atomKey.split("(")[0]
                if not classKey.startswith("C"):
                    classKey = "C_" + classKey
                coreValues, latency = ftoc.factDf_to_class_values(df,
                                                                  individuals=individualsDict[indKey],
                                                                  classes=candidatesDict[atomKey],
                                                                  prefix=prefix)
                self.fixedCoresDict[atomKey] = cc.CoordinateCore(coreValues, [indKey, classKey])

    def generate_targetCore_pairDf(self, pairDf, individualsDict,targetIsFilter=True):
        skeletonIndividuals = ec.get_individuals(self.skeleton)
        coreValues, latency = ptoc.pairDf_to_target_values(pairDf, individualsDict, skeletonIndividuals)
        self.targetCore = cc.CoordinateCore(coreValues, skeletonIndividuals)
        if targetIsFilter:
            self.filterCore = self.targetCore.clone()

    def random_initialize_variableCoresDict(self):
        self.variablesCoresDict = {}
        for legKey in self.fixedCoresDict:
            varColor, varDim = variable_find_var(self.fixedCoresDict[legKey])
            self.variablesCoresDict[legKey] = cc.CoordinateCore(np.random.random(varDim), [varColor])


def atomic_find_var(fixedCore):
    varColors = []
    varDims = []
    for i, color in enumerate(fixedCore.colors):
        if color != "j":
            varColors.append(color)
            varDims.append(fixedCore.values.shape[i])
    return varColors, varDims


def variable_find_var(fixedCore):
    varColors = []
    for color in fixedCore.colors:
        if color.startswith("C") or color.startswith("R"):
            varColors.append(color)
    if len(varColors) != 1:
        raise TypeError("Variables in Core with colors {} not found correctly!".format(fixedCore.colors))
    return varColors[0], fixedCore.values.shape[fixedCore.colors.index(varColors[0])]


if __name__ == "__main__":
    import pandas as pd

    samDf = pd.read_csv("./examples/generation/synthetic_test_data/generated_sampleDf.csv")

    skeleton = ["P0", "and", "P1"]
    candidatesDict = {
        "P0": list(samDf.columns),
        "P1": list(samDf.columns)
    }
    learner = AtomicLearner(skeleton)
    learner.generate_fixedCores_sampleDf(samDf, candidatesDict)

    learner.random_initialize_variableCoresDict()

    learner.set_targetCore(length=samDf.values.shape[0])
    learner.als(2)

    learner.get_solution()
    print(learner.solutionExpression)
