import pandas as pd
import numpy as np

from tnreason.model import generate_test_data as gtd
from tnreason.learning import expression_learning as el

from tnreason.logic import expression_calculus as ec

## DATA GENERATION ##

## Specify parameters
# example_expression_dict specifies an Markov Logic Network: keys are arbitrary, values contain formula and weights
# dataNum is the number of samples to be generated
# regenerate whether data is instead loaded from the synthetic_test_data folder
example_expression_dict = {
    "e0": ["Rechnung(x)", 1.5],
    "e1": [["not", "Ausgangsrechnung(x)"], 1.5],
    "e2": [["Ausgangsrechnung(x)", "and", "zuMandant(x,y)"], 2],
    "e3": [[["not", "Ausgangsrechnung(y)"], "and", ["not", "Rechnung(x)"]], 2],
    "e4": [["Ausgangsrechnung(x)", "and", "Rechnung(y)"], 2],
    "e5": ["Spezialrechnung(x)", 1.5],
    "e6": [["versandt(y,x)", "and", "Ausgangsrechnung(x)"], 2],
    "e7": [["bearbeitet(y,x)", "and", "enthaelt(x,z)"], 1.2]
}
dataNum = 10
regenerate = True
savePath = "./demonstration/learning/synthetic_test_data/synthetic_accounting/"
if regenerate:
    factDf, pairDf = gtd.generate_factDf_and_pairDf(example_expression_dict, sampleNum=dataNum, prefix="")
    factDf.to_csv(savePath + "generated_factDf.csv")
    pairDf.to_csv(savePath + "generatedfr_pairDf.csv")
else:
    factDf = pd.read_csv(savePath + "generated_factDf.csv", index_col=0)
    pairDf = pd.read_csv(savePath + "generated_pairDf.csv", index_col=0)

## Expression Learning ##

## Specify the Individuals to be represented on the axes
# keys of individualsDict correspond with colors of the respective axes
# values of individualsDict correspond with interpretation of the coordinates on the respective axes
individualsDict = {
    "x": np.unique(pairDf["x"].values),
    "y": np.unique(pairDf["y"].values),
    "z": np.unique(pairDf["z"].values)
}

## Define the search space for the solution expression
# skeletonExpression provides the overall structure of the expression with placeholders
# candidatesDict provides the possible atoms to be plugged in the placeholder
skeletonExpression = ["R1(y,x)", "and", "C1(x)"]  # ,"and","R2(x,z)"]
candidatesDict = {
    "C1(x)": ["Ausgangsrechnung(x)", "Rechnung(x)"],
    "C2(y)": ["Ausgangsrechnung(y)", "Rechnung(y)"],
    "R1(y,x)": ["versandt(y,x)", "bearbeitet(y,x)"],
    "R2(x,z)": ["enthaelt(x,z)"]
}
## Initialize the ExpressionLearner and generate all Tensor Cores
# fixedCores based on the factDf
# targetCore and filterCore based on the pairDf
# variableCores from random
learner = el.VariableLearner(skeletonExpression=skeletonExpression)
#learner = el.ExpressionLearner(skeletonExpression=skeletonExpression)
learner.generate_fixedCores_factDf(factDf, individualsDict, candidatesDict, prefix="")

positiveExpression = ["versandt(y,x)","and","Rechnung(x)"]
positiveCore = ec.evaluate_expression_on_factDf(factDf,individualsDict,positiveExpression)

negativeExpression = ["bearbeitet(y,x)","and","Ausgangsrechnung(x)"]
negativeCore = ec.evaluate_expression_on_factDf(factDf,individualsDict,negativeExpression)

learner.generate_target_and_filterCore_from_exampleCores(positiveCore,negativeCore)

learner.random_initialize_variableCoresDict()

## Optimize the variableCores using Alternating Least Squares and create the solution based on the largest core values
learner.als(10)
learner.get_solution()
print("#####")
print("The solution expression is:")
print(learner.solutionExpression)

solutionCore = ec.evaluate_expression_on_factDf(factDf,individualsDict,learner.solutionExpression)
print("with residuum norm:  {}".format(np.linalg.norm(solutionCore.values - learner.targetCore.values)))
