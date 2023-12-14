import numpy as np

from tnreason.model import generate_test_data as gtd

from tnreason.logic import expression_generation as eg

from tnreason.learning import mln_learning as mlnl

from tnreason.contraction import expression_evaluation as ee

from benchmarking import recovery_experiment as re

import matplotlib.pyplot as plt

formulaweight = 2
formulaNum = 3
sampleNum = 100

example_rule_dict = {"e"+str(i):  [["versandterBeleg(y,x)"+str(i),'hatLeistungserbringer(x,y)'+str(i)],"Ausgangsrechnung(x)", formulaweight]
                     for i in range(formulaNum)}
formulaDict = {key: [eg.generate_list_from_rule(value[0], value[1]), value[2]] for (key, value) in
           example_rule_dict.items()}


decoupling_strength = 4
decoupling_dict = {}
for i in range(formulaNum):
    for j in range(formulaNum):
        decoupling_dict["d"+str(i)+str(j)] = [["not",[["versandterBeleg(y,x)"+str(i), "and", 'hatLeistungserbringer(x,y)'+str(i)]
            ,"and",["versandterBeleg(y,x)"+str(j), "and", 'hatLeistungserbringer(x,y)'+str(j)]]], decoupling_strength]

formulaDict = {**formulaDict, **decoupling_dict}

positiveExpression = "Ausgangsrechnung(x)"
skeletonExpression = ["P1", "and", "P2"]
candidatesDict = {
    "P1": ["versandterBeleg(y,x)"+str(i) for i in range(formulaNum)],
    "P2": ["hatLeistungserbringer(x,y)"+str(i) for i in range(formulaNum)],
}

sampleDf = gtd.generate_sampleDf(formulaDict, sampleNum= sampleNum, method="Gibbs3").astype("int64")
print(sampleDf.head())

for formulaKey in formulaDict:
    expression = formulaDict[formulaKey][0]
    print(expression)

    print(ee.ExpressionEvaluator(expression).evaluate_on_sampleDf(sampleDf).count_satisfaction())

print(ee.ExpressionEvaluator("Ausgangsrechnung(x)").evaluate_on_sampleDf(sampleDf).count_satisfaction())
print(ee.ExpressionEvaluator(["versandterBeleg(y,x)0","and","hatLeistungserbringer(x,y)0"]).evaluate_on_sampleDf(sampleDf).count_satisfaction())

learner = mlnl.SampleBasedMLNLearner()
learner.load_sampleDf(sampleDf)

learner.learn("Ausgangsrechnung(x)", skeletonExpression, candidatesDict,
              boostNum=formulaNum, saveMod="imp", refinementNum=0,acceptanceCriterion="weight>0")