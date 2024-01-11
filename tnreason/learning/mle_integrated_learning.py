## Integrates the former separated weight and formula learning
from tnreason.optimization import alternating_mle as amle
from tnreason.optimization import weight_estimation as wees


class FormulaLearner:
    def __init__(self, learnedFormulaDict={}):
        self.learnedFormulaDict = learnedFormulaDict
        self.weightEstimator = wees.WeightEstimator(
            formulaList=[learnedFormulaDict[key][0] for key in learnedFormulaDict],
            startWeightsDict=[learnedFormulaDict[key][0] for key in learnedFormulaDict])

    def learn(self, skeletonExpression, candidatesDict, sampleDf):
        optimizer = amle.GradientDescentMLE(skeletonExpression, candidatesDict,
                                            learnedFormulaDict=self.learnedFormulaDict, sampleDf=sampleDf)
        optimizer.alternating_gradient_descent(sweepNum=100, stepWidth=1)
        solutionExpression = optimizer.get_solution_expression()

        ## To be implemented! Check in mln_learning for similarity.
        self.weightEstimator.add_formula("newFormula", solutionExpression, weight=0)
