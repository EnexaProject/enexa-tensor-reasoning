from tnreason.learning import mle_integrated_learning as mlel
from tnreason.model import generate_test_data as gtd

learnedFormulaDict = {
    "f0": ["a2", 1.2],
    "f1": [["a1", "and", "a2"], 1.3],
    "f2": ["a3", 2],
}

skeletonExpression = ["not",["P1", "and", ["not","P2"]]]

candidatesDict = {"P1": ["a1", "a2", "a3"],
                  "P2": ["a1", "a2", "a3"],
                  }

sampleDf = gtd.generate_sampleDf(learnedFormulaDict, 100)

learner = mlel.GibbsFormulaLearner(knownFormulasDict={"fun":[["a3","and","a3"],2]},
                              sampleDf=sampleDf)
learner.learn(skeletonExpression, candidatesDict, "fun2")

hybridKB = learner.to_hybrid_kb()
print(hybridKB.ask(["a1","and","a2"]))


learner2 = mlel.GDFormulaLearner(knownFormulasDict={"fun":[["a3","and","a3"],2]},
                              sampleDf=sampleDf)


learner2.learn(skeletonExpression, candidatesDict, "learned1")

skeletonExpression = "P1"
candidatesDict = {"P1": ["a1", "a2", "a3"]
                  }
learner2.learn(skeletonExpression, candidatesDict, "learned2")
learner2.learn(skeletonExpression, candidatesDict, "learned3")


learner2.adjust_weights()
print(learner2.get_learned_formulas())
