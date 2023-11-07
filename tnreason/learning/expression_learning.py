import tnreason.representation.factdf_to_cores as ftoc
import tnreason.representation.sampledf_to_cores as stoc
import tnreason.representation.pairdf_to_cores as ptoc

import tnreason.logic.expression_calculus as ec
import tnreason.logic.coordinate_calculus as cc
import tnreason.logic.expression_generation as eg

import tnreason.optimization.generalized_als as gals

import tnreason.model.create_mln as cmln

import numpy as np


class ExpressionLearner:
    def __init__(self, skeletonExpression):
        self.skeleton = skeletonExpression
        self.skeletonAtoms = ec.get_variables(skeletonExpression)
        self.skeletonIndividuals = ec.get_individuals(skeletonExpression)

        self.targetCore = None
        self.filterCore = None

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

    def generate_fixedCores_turtlePath(self, turtlePath, limit, individualsDict, candidatesDict):
        self.generate_fixedCores_factDf(ftoc.generate_factDf(turtlePath, limit), individualsDict, candidatesDict)

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

    def random_initialize_variableCoresDict(self):
        self.variablesCoresDict = {}
        for legKey in self.fixedCoresDict:
            varColor, varDim = find_var(self.fixedCoresDict[legKey])
            self.variablesCoresDict[legKey] = cc.CoordinateCore(np.random.random(varDim), [varColor])

    def generate_targetCore_pairDf(self, pairDf, individualsDict):
        coreValues, latency = ptoc.pairDf_to_target_values(pairDf, individualsDict, self.skeletonIndividuals)
        self.targetCore = cc.CoordinateCore(coreValues, self.skeletonIndividuals)

    def set_targetCore(self, targetCore, targetIsFilter=True):
        self.targetCore = targetCore
        if targetIsFilter:
            self.set_filterCore(targetCore)

    def set_filterCore(self, filterCore):
        self.filterCore = filterCore

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

        #self.optimizer.visualize_residua()

    def get_solution(self):
        self.solutionDict = {}
        for legKey in self.variablesCoresDict:
            maxPos = np.argmax([abs(val) for val in self.variablesCoresDict[legKey].values])
            self.solutionDict[legKey] = self.candidatesDict[legKey][maxPos] + "(" + legKey.split("(")[1]
        self.solutionExpression = eg.replace_atoms(self.skeleton, self.solutionDict)

    @property
    def factor_core(self):
        return cmln.calculate_dangling_basis(self.solutionExpression).calculate_truth()


def find_var(fixedCore):
    varColors = []
    for color in fixedCore.colors:
        if color.startswith("C") or color.startswith("R"):
            varColors.append(color)
    if len(varColors) != 1:
        raise TypeError("Variables in Core with colors {} not found correctly!".format(fixedCore.colors))
    return varColors[0], fixedCore.values.shape[fixedCore.colors.index(varColors[0])]


if __name__ == "__main__":
    skeleton = [["C1(a)", "and", "R1(a,b)"], "and", ["not", "C1(a)"]]
    el = ExpressionLearner(skeleton)

    individualsDict = {
        "a": ["http://datev.de/ontology#ocr_item_5f3d5dc5-c55e-d6bc-50bf-7071e0f90d61_6"],
        "b": ["http://datev.de/ontology#ocr_item_5f3d5dc5-c55e-d6bc-50bf-7071e0f90d61_6"]
    }

    candidatesDict = {
        "C1(a)": ["http://datev.de/ontology#OcrItemsSubClass_65504"],
        "R1(a,b)": ["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]
    }

    import pandas as pd

    df = pd.read_csv("./results/csv_files/parquet_tev2_list.csv", index_col=0)

    el.generate_fixedCores_factDf(df, individualsDict, candidatesDict)

    el.random_initialize_variableCoresDict()

    targetCore = cc.CoordinateCore(np.random.normal(size=(1, 1)), ["a", "b"])
    el.set_targetCore(targetCore)
    el.als(sweepnum=1)

    el.get_solution()

    print(el.factor_core.values.shape)

    print(el.solutionDict)
    print(el.solutionExpression)
