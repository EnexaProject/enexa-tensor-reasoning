import tnreason.logic.expression_calculus as ec
import tnreason.logic.coordinate_calculus as cc

import tnreason.representation.sampledf_to_cores as stoc

import tnreason.optimization.generalized_als as gals

import numpy as np

class GenericExpressionLearner:
    def __init__(self, skeletonExpression):
        self.skeleton = skeletonExpression
        self.skeletonAtoms = ec.get_variables(skeletonExpression)

        self.targetCore = None
        self.filterCore = None

    def generate_fixedCores_sampleDf(self, sampleDf, candidatesDict):

        self.fixedCoresDict = {}

        for atomKey in self.skeletonAtoms:
            values = stoc.sampleDf_to_universal_core(sampleDf,candidatesDict[atomKey])
            self.fixedCoresDict[atomKey] = cc.CoordinateCore(values,["j",atomKey])

    def random_initialize_variableCoresDict(self):
        self.variablesCoresDict = {}
        for legKey in self.fixedCoresDict:
            varColors, varDims = find_var(self.fixedCoresDict[legKey])
            self.variablesCoresDict[legKey] = cc.CoordinateCore(np.random.random(varDims), varColors)

    def set_targetCore(self, targetCore=None, length=0, targetIsFilter=True):
        if targetCore is None:
            self.targetCore = cc.CoordinateCore(np.ones([length]),["j"])
        else:
            self.targetCore = targetCore

        if targetIsFilter:
            self.filterCore = self.targetCore.clone()

    def als(self, sweepnum):
        self.optimizer = gals.GeneralizedALS(self.variablesCoresDict, self.fixedCoresDict)
        self.optimizer.set_targetCore(self.targetCore)
        self.optimizer.set_filterCore(self.filterCore)

        self.optimizer.sweep(sweepnum=sweepnum, contractionScheme=self.skeleton)
        self.variablesCoresDict = self.optimizer.variableCoresDict


def find_var(fixedCore):
    varColors = []
    varDims = []
    for i, color in enumerate(fixedCore.colors):
        if color != "j":
            varColors.append(color)
            varDims.append(fixedCore.values.shape[i])
    return varColors, varDims

if __name__ == "__main__":
    import pandas as pd
    samDf = pd.read_csv("./examples/generation/synthetic_test_data/generated_sampleDf.csv")

    skeleton = ["P0","and","P1"]
    candidatesDict = {
        "P0" : list(samDf.columns),
        "P1" : list(samDf.columns)
    }
    learner = GenericExpressionLearner(skeleton)
    learner.generate_fixedCores_sampleDf(samDf, candidatesDict)


    learner.random_initialize_variableCoresDict()
    print("## BEFORE")
    print(learner.variablesCoresDict["P0"].values)
    print(learner.variablesCoresDict["P1"].values)

    learner.set_targetCore(length=samDf.values.shape[0])
    learner.als(2)


    print("## END")
    print(learner.variablesCoresDict["P0"].values)
    print(learner.variablesCoresDict["P1"].values)