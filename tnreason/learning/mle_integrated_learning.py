## Integrates the former separated weight and formula learning
from tnreason.optimization import alternating_mle as amle
from tnreason.optimization import weight_estimation as wees

from tnreason.learning import formula_sampling as fs


class FormulaLearnerBase:
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

    def add_formula(self, formula, formulaKey, verbose=True):
        self.weightEstimator.add_formula(formulaKey, formula, weight=0)
        self.weightEstimator.formula_optimization(formulaKey)
        assignedWeight = self.weightEstimator.formulaDict[formulaKey][3]
        self.learnedFormulaDict[formulaKey] = [formula, assignedWeight]
        if verbose:
            print("The formula {} has been added with weight {}.".format(formula, assignedWeight))

    def adjust_weights(self, sweepNum=10):
        self.weightEstimator.alternating_optimization(sweepNum)
        weightDict = self.weightEstimator.get_weights()
        for key in self.learnedFormulaDict:
            self.learnedFormulaDict[key][1] = weightDict[key]

    def get_learned_formulas(self):
        return self.learnedFormulaDict


class GDFormulaLearner(FormulaLearnerBase):
    def learn(self, skeletonExpression, candidatesDict, formulaKey=None, gdSweepNum=10):
        optimizer = amle.GradientDescentMLE(skeletonExpression, candidatesDict,
                                            learnedFormulaDict=self.learnedFormulaDict,
                                            sampleDf=self.sampleDf)
        optimizer.alternating_gradient_descent(sweepNum=gdSweepNum, stepWidth=1)
        solutionExpression = optimizer.get_solution_expression()

        if formulaKey is None:
            formulaKey = str(solutionExpression)

        self.add_formula(solutionExpression, formulaKey)


class GibbsFormulaLearner(FormulaLearnerBase):
    def learn(self, skeletonExpression, candidatesDict, formulaKey=None,
              annealingPattern=[(10, 2), (10, 1), (10, 0.5), (10, 0.2), (10, 0.1)]):
        learner = fs.GibbsFormulaSampler(skeletonExpression, candidatesDict, self.learnedFormulaDict,
                                         sampleDf=self.sampleDf)
        learner.gibbs_simulated_annealing(annealingPattern)

        learnedFormula = learner.get_result()
        self.add_formula(learnedFormula, formulaKey)
