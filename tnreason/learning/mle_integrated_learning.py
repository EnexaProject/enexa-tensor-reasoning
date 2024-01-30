## Integrates the former separated weight and formula learning
from tnreason.optimization import alternating_mle as amle
from tnreason.optimization import entropy_maximization as entm

from tnreason.learning import formula_sampling as fs


class FormulaLearnerBase:
    def __init__(self, knownFormulasDict={}, knownFactsDict={}, sampleDf=None):

        self.learnedFormulaDict = knownFormulasDict.copy()
        self.knownFactsDict = knownFactsDict.copy()

        self.entropyMaximizer = entm.EntropyMaximizer(formulaDict=knownFormulasDict.copy(),
                                                      factDict=knownFactsDict.copy())
        self.load_sampleDf(sampleDf)

    def load_sampleDf(self, sampleDf):
        self.sampleDf = sampleDf
        self.entropyMaximizer.calculate_satisfaction(sampleDf)

    def add_formula(self, formula, formulaKey, verbose=True, weightThreshold=0, adjustSweeps=0):
        self.entropyMaximizer.add_formula(formula, key=formulaKey, isFact=False, sampleDf=self.sampleDf)
        self.entropyMaximizer.formula_optimization(formulaKey)
        if adjustSweeps > 0:
            self.adjust_weights(adjustSweeps)
        assignedWeight = self.entropyMaximizer.get_weights([formulaKey])[formulaKey]

        if assignedWeight > weightThreshold:
            self.learnedFormulaDict[formulaKey] = [formula, assignedWeight]
            if verbose:
                print("The formula {} has been added with weight {}.".format(formula, assignedWeight))
        else:
            if verbose:
                print("The formula {} has not been added since the weight {} does not exceed the threshold {}.".format(
                    formula, assignedWeight, weightThreshold))
            self.entropyMaximizer.drop_formula(formulaKey)

    def adjust_weights(self, sweepNum=10):
        self.entropyMaximizer.alternating_optimization(sweepNum)
        weightDict = self.entropyMaximizer.get_weights()
        for key in weightDict:
            self.learnedFormulaDict[key][1] = weightDict[key]

    def get_learned_formulas(self):
        return self.learnedFormulaDict


class GDFormulaLearner(FormulaLearnerBase):
    ## Does not make use of the constants so far! Not supported yet in amle
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
        learner = fs.GibbsFormulaSampler(skeletonExpression, candidatesDict,
                                         knownFormulasDict=self.learnedFormulaDict,
                                         knownFactsDict=self.knownFactsDict,
                                         sampleDf=self.sampleDf)
        learner.gibbs_simulated_annealing(annealingPattern)

        learnedFormula = learner.get_result()
        self.add_formula(learnedFormula, formulaKey)
