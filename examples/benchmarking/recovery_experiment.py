import pandas as pd
import numpy as np

from tnreason.model import generate_test_data as gtd
from tnreason.learning import expression_learning as el

from tnreason.logic import expression_calculus as ec


def generate_data(formulaDict, sampleNum):
    return gtd.generate_sampleDf(formulaDict, sampleNum=sampleNum, chainSize=10)


def sampleDf_experiment(formulaDict, sampleNum, skeletonExpression, candidatesDict, positiveExpression,
                        trueExpression=None):
    if trueExpression is None:
        trueExpression = positiveExpression

    sampleDf = generate_data(formulaDict, sampleNum).astype("int64")

    learner = el.AtomicLearner(skeletonExpression)
    learner.generate_fixedCores_sampleDf(sampleDf, candidatesDict)
    learner.random_initialize_variableCoresDict()

    positiveCore = ec.evaluate_expression_on_sampleDf(sampleDf, positiveExpression)
    negativeCore = positiveCore.negate()

    learner.generate_target_and_filterCore_from_exampleCores(positiveCore, negativeCore)

    learner.als(10)
    learner.get_solution()
    solutionExpression = learner.solutionExpression

    if solutionExpression == trueExpression:
        return 1
    else:
        return 0
