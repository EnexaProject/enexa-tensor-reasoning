from tnreason.learning import mle_integrated_learning as mlel

from tnreason.model import generate_test_data as gtd

sampleDf = gtd.generate_sampleDf(
    {
        "e0" : [["a1","xor","a2"], 3],
        "e1" : [["targetAtom1","and","targetAtom2"], 2]
    },
    sampleNum=100
)


learner = mlel.GibbsFormulaLearner(knownFormulasDict={},
                         knownFactsDict={"fact1":"a1"},
                        sampleDf=sampleDf)
smallSkeletonExpression = ["P1","xor","P2"]
positiveExpression = ["Y1","and","Y2"]
learner.learn(skeletonExpression=[smallSkeletonExpression,"eq",positiveExpression],
              candidatesDict= {"P1": ["a1","a2"],
                               "P2": ["a1","a2"],
                               "Y1" : ["targetAtom1"],
                               "Y2" : ["targetAtom2"]},

              )
learner.adjust_weights(10)