from tnreason.learning import mle_integrated_learning as mlel
from tnreason.model import generate_test_data as gtd

learnedFormulaDict = {
    "f0": ["a2", 1.2],
    "f1": [["a1", "and", "a2"], 1.3],
    "f2": ["a3", 2]
}
sampleDf = gtd.generate_sampleDf(learnedFormulaDict, 10)

learner = mlel.FormulaLearner()

skeletonExpression = ["P1", "and", "P2"]
candidatesDict = {"P1": ["a1","a2","a3"],
                  "P2": ["a1","a2","a3"],
                  }


learner.learn(skeletonExpression, candidatesDict, sampleDf)