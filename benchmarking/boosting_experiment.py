import numpy as np

from tnreason.model import generate_test_data as gtd

from tnreason.logic import expression_generation as eg

from tnreason.learning import mln_learning as mlnl

from tnreason.contraction import expression_evaluation as ee

from benchmarking import recovery_experiment as re

import matplotlib.pyplot as plt


def boost_experiment(formulaweight=2, formulaNum=3, sampleNum=100, evaluate_truths=False):
    firstAtom = "versandterBeleg(y,x)"
    secondAtom = "hatLeistungserbringer(x,y)"
    positiveExpression = "Ausgangsrechnung(x)"

    example_rule_dict = {"e" + str(i): [[firstAtom + str(i), secondAtom + str(i)], "Ausgangsrechnung(x)", formulaweight]
                         for i in range(formulaNum)}
    rawDict = {key: [eg.generate_list_from_rule(value[0], value[1]), value[2]] for (key, value) in
               example_rule_dict.items()}

    decoupling_strength = 4
    decoupling_dict = {}
    for i in range(formulaNum):
        for j in range(formulaNum):
            decoupling_dict["d" + str(i) + str(j)] = [["not", [[firstAtom + str(i), "and", secondAtom + str(i)]
                , "and", [firstAtom + str(j), "and", secondAtom + str(j)]]], decoupling_strength]

    formulaDict = {**rawDict, **decoupling_dict}

    skeletonExpression = ["P1", "and", "P2"]
    candidatesDict = {
        "P1": [firstAtom + str(i) for i in range(formulaNum)],
        "P2": [secondAtom + str(i) for i in range(formulaNum)],
    }

    sampleDf = gtd.generate_sampleDf(formulaDict, sampleNum=sampleNum, method="Gibbs3").astype("int64")

    if evaluate_truths:
        for formulaKey in formulaDict:
            expression = formulaDict[formulaKey][0]
            print(expression)
            print(ee.ExpressionEvaluator(expression).evaluate_on_sampleDf(sampleDf).count_satisfaction())

    learner = mlnl.SampleBasedMLNLearner()
    learner.load_sampleDf(sampleDf)

    learner.learn(positiveExpression, skeletonExpression, candidatesDict,
                  boostNum=formulaNum, saveMod="imp", refinementNum=0,
                  acceptanceCriterion="weight>" + str(formulaweight *  (2/3)))

    learned_formulas = [learner.weightedFormulas[i][0] for i in range(len(learner.weightedFormulas))]
    learned_pairs = [[formula[1][0][0].split(firstAtom)[1], formula[1][0][2].split(secondAtom)[1]] for formula in
                     learned_formulas]

    trueCount = 0
    for i in range(formulaNum):
        if [str(i), str(i)] in learned_pairs:
            trueCount += 1
    falseCount = 0
    for a, b in learned_pairs:
        if a != b:
            falseCount += 1

    return len(learned_pairs), trueCount, falseCount


if __name__ == "__main__":
    formulaNum = 3
    formulaWeight = 2
    sampleNums = range(50, 1000, 50)

    learnshape = len(sampleNums)
    learnedCounts = np.empty(shape=learnshape)
    trueCounts = np.empty(shape=learnshape)
    falseCounts = np.empty(shape=learnshape)

    for i, sampleNum in enumerate(sampleNums):
        learnedCount, trueCount, falseCount = boost_experiment(formulaweight=formulaWeight, formulaNum=formulaNum,
                                                               sampleNum=sampleNum)
        print(sampleNum, learnedCount, trueCount, falseCount)

        learnedCounts[i] = learnedCount
        trueCounts[i] = trueCount
        falseCounts[i] = falseCount

    from matplotlib import pyplot as plt

    plt.scatter(range(learnshape), learnedCounts, color="r", marker="+", label="Totally Learned")
    plt.scatter(range(learnshape), trueCounts, color="b", marker="+", label="True Learned")
    plt.scatter(range(learnshape), falseCounts, color="g", marker="+", label="False Learned")

    plt.legend()
    plt.xticks(range(learnshape), [str(num) for num in sampleNums])
    plt.yticks(range(formulaNum+1), [str(num) for num in range(formulaNum+1)])
    plt.ylim([0, formulaNum+1])

    plt.ylabel("Number of Formulas")
    plt.xlabel("Number of Samples")
    plt.title("Recovery Rate when having 3 Rules with Same Head")
    plt.savefig("./benchmarking/diagrams/boosting_recoveries_weight{}.png".format(formulaWeight))

    plt.show()


