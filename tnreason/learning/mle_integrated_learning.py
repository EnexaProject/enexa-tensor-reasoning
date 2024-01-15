## Integrates the former separated weight and formula learning
from tnreason.optimization import alternating_mle as amle
from tnreason.optimization import weight_estimation as wees


class FormulaLearner:
    def __init__(self, learnedFormulaDict={}, sampleDf=None):
        self.learnedFormulaDict = learnedFormulaDict
        self.weightEstimator = wees.WeightEstimator(sampleDf=sampleDf)
        for formulaKey in learnedFormulaDict:
            self.weightEstimator.add_formula(formulaKey, learnedFormulaDict[formulaKey][0],
                                             learnedFormulaDict[formulaKey][1])
        self.load_sampleDf(sampleDf)

    def load_sampleDf(self, sampleDf):
        self.sampleDf = sampleDf
        self.weightEstimator.sampleDf = sampleDf

    def learn(self, skeletonExpression, candidatesDict, formulaKey=None, gdSweepNum=10):
        ## Step1: Formula Learning
        ## For each Formula Learning Step initialize an optimizer
        optimizer = amle.GradientDescentMLE(skeletonExpression, candidatesDict,
                                            learnedFormulaDict=self.learnedFormulaDict,
                                            sampleDf=self.sampleDf)
        optimizer.alternating_gradient_descent(sweepNum=gdSweepNum, stepWidth=1)
        solutionExpression = optimizer.get_solution_expression()

        if formulaKey is None:
            formulaKey = str(solutionExpression)

        ## Step2: Weight Adjusting
        self.weightEstimator.add_formula(formulaKey, solutionExpression, weight=0)
        self.weightEstimator.formula_optimization(formulaKey)
        assignedWeight = self.weightEstimator.formulaDict[formulaKey][3]
        self.learnedFormulaDict[formulaKey] = [solutionExpression, assignedWeight]

        print("The formula {} has been added with weight {}.".format(solutionExpression, assignedWeight))

    def adjust_weights(self, sweepNum=10):
        self.weightEstimator.alternating_optimization(sweepNum)
        weightDict = self.weightEstimator.get_weights()
        for key in self.learnedFormulaDict:
            self.learnedFormulaDict[key][1] = weightDict[key]

    def get_learned_formulas(self):
        return self.learnedFormulaDict
