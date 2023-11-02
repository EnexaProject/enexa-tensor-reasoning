import pandas as pd
import numpy as np

from tensor_reasoning.generation import generate_test_data as gtd

from tensor_reasoning.learning import expression_learning as el

## DATA GENERATION ##

## Specify parameters
# example_expression_dict specifies an Markov Logic Network: keys are arbitrary, values contain formula and weights
# dataNum is the number of samples to be generated
# regenerate whether data is instead loaded from the synthetic_test_data folder
weight = 1.5
example_expression_dict = {
    "a": [["holdsAt(x,t1)", "and", "holdsAt(x,t2)"], 10],
    "t1": [["not", ["terminatedAt(x,t1)", "and", "holdsAt(x,t1)"]], weight],
    "t2": [["not", ["terminatedAt(x,t2)", "and", "holdsAt(x,t2)"]], weight],
    "i1": [["not", ["initiatedAt(x,t1)", "and", ["not", "holdsAt(x,t1)"]]], weight],
    "i2": [["not", ["initiatedAt(x,t2)", "and", ["not", "holdsAt(x,t2)"]]], weight],
    "e12": [["not", [[["not", "initiatedAt(x,t2)"], "and", "terminatedAt(x,t1)"], "and", "holdsAt(x,t2)"]], weight],
    "i12": [["not", ["holdsAt(x,t1)", "and", [["not", "holdsAt(x,t2)"], "and", ["not", "terminatedAt(x,t2)"]]]], weight]
}
dataNum = 10
regenerate = True
savePath = "./examples/synthetic_test_data/learning/synthetic_ec/"
if regenerate:
    factDf, pairDf = gtd.generate_factDf_and_pairDf(example_expression_dict, sampleNum=dataNum, prefix="")
    factDf.to_csv(savePath + "generated_factDf.csv")
    pairDf.to_csv(savePath + "generated_pairDf.csv")
else:
    factDf = pd.read_csv(savePath + "generated_factDf.csv", index_col=0)
    pairDf = pd.read_csv(savePath + "generated_pairDf.csv", index_col=0)

## Expression Learning ##

## Specify the Individuals to be represented on the axes
# keys of individualsDict correspond with colors of the respective axes
# values of individualsDict correspond with interpretation of the coordinates on the respective axes
individualsDict = {
    "x": np.unique(pairDf["x"].values),
    "t1": np.unique(pairDf["t1"].values),
    "t2": np.unique(pairDf["t2"].values),
}

## Define the search space for the solution expression
# skeletonExpression provides the overall structure of the expression with placeholders
# candidatesDict provides the possible atoms to be plugged in the placeholder
skeletonExpression = ["R1(x,t1)", "and", "R2(x,t2)"]
candidatesDict = {
    "R1(x,t1)": ["holdsAt", "terminatedAt", "initiatedAt"],
    "R2(x,t2)": ["holdsAt", "terminatedAt", "initiatedAt"]
}

## Initialize the ExpressionLearner and generate all Tensor Cores
# fixedCores based on the factDf
# targetCore and filterCore based on the pairDf
# variableCores from random
learner = el.ExpressionLearner(skeletonExpression=skeletonExpression)
learner.generate_fixedCores_factDf(factDf, individualsDict, candidatesDict, prefix="")
learner.generate_targetCore_pairDf(pairDf, individualsDict)
learner.set_filterCore(learner.targetCore)
learner.random_initialize_variableCoresDict()

## Optimize the variableCores using Alternating Least Squares and create the solution based on the largest core values
learner.als(3)

learner.get_solution()
print("The solution expression is:")
print(learner.solutionExpression)
