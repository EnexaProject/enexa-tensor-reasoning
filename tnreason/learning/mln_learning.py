import tnreason.logic.expression_generation as eg

import tnreason.optimization.weight_estimation as wees
import tnreason.optimization.expression_refinement as er

import tnreason.contraction.expression_evaluation as ee

import tnreason.learning.expression_learning as el

from tnreason.model import tensor_network_mln as mln


class SampleBasedMLNLearner:
    def __init__(self, sampleDf=None):
        self.weightedFormulas = []
        if sampleDf is not None:
            self.load_sampleDf(sampleDf)

    def load_sampleDf(self, sampleDf):
        self.sampleDf = sampleDf

    def learn_implication(self, positiveExpression, skeletonExpression, candidatesDict,
                          refinementNum=0, refinementCriterion="weight>1", acceptanceCriterion="weight>0"):
        self.learn(positiveExpression, skeletonExpression, candidatesDict, boostNum=1, saveMod="imp",
                   refinementNum=refinementNum,
                   refinementCriterion=refinementCriterion,
                   acceptanceCriterion=acceptanceCriterion
                   )

    def learn_equivalence(self, positiveExpression, skeletonExpression, candidatesDict,
                          refinementNum=0, refinementCriterion="weight>1", acceptanceCriterion="weight>0"):
        self.learn(positiveExpression, skeletonExpression, candidatesDict, boostNum=1, saveMod="eq",
                   refinementNum=refinementNum,
                   refinementCriterion=refinementCriterion,
                   acceptanceCriterion=acceptanceCriterion
                   )

    def learn_tautology(self, skeletonExpression, candidatesDict,
                        refinementNum=0, refinementCriterion="weight>1", acceptanceCriterion="weight>0"):
        self.learn("Thing", skeletonExpression, candidatesDict, boostNum=1, saveMod="direct",
                   refinementNum=refinementNum,
                   refinementCriterion=refinementCriterion,
                   acceptanceCriterion=acceptanceCriterion
                   )

    def learn(self, positiveExpression, skeletonExpression, candidatesDict,
              boostNum=1, saveMod="eq",
              refinementNum=0, refinementCriterion="weight>1",
              balance=True,
              acceptanceCriterion="weight>0"):
        if saveMod == "eq" or saveMod == "imp":
            # positiveCore = ec.evaluate_expression_on_sampleDf(self.sampleDf, positiveExpression)
            positiveCore = ee.ExpressionEvaluator(positiveExpression, sampleDf=self.sampleDf).evaluate()
        else:
            # positiveCore = ec.evaluate_expression_on_sampleDf(self.sampleDf, "Thing")
            positiveCore = ee.ExpressionEvaluator("Thing", sampleDf=self.sampleDf).evaluate()

        solutionExpressions = self.boost(skeletonExpression, candidatesDict, positiveCore, boostNum=boostNum,
                                         refinementNum=refinementNum, refinementCriterion=refinementCriterion,
                                         balance=balance)
        ## Adding the formulas to the model
        for learnedPremise in solutionExpressions:
            if saveMod == "eq" or saveMod == "imp":
                solutionExpression = eg.generate_from_generic_expression([learnedPremise, saveMod, positiveExpression])
                self.add_independent_formula(solutionExpression, criterion=acceptanceCriterion)
            else:
                self.add_independent_formula(learnedPremise, criterion=acceptanceCriterion)

    def boost(self, skeletonExpression, candidatesDict, positiveCore, boostNum=1, refinementNum=0,
              refinementCriterion="weight>1",
              balance=True):
        solutionExpressions = []
        for boostPos in range(boostNum):
            print("## Boost number {} ##".format(boostPos))
            refiningExpression = "Thing"
            for solutionExpression in solutionExpressions:
                refiningExpression = [refiningExpression, "and", ["not", solutionExpression]]
            # refiningCore = ec.evaluate_expression_on_sampleDf(self.sampleDf, refiningExpression)
            refiningCore = ee.ExpressionEvaluator(refiningExpression, sampleDf=self.sampleDf).evaluate()
            positiveCore = positiveCore.compute_and(refiningCore)
            negativeCore = positiveCore.negate()

            solutionExpressions.append(self.refine(skeletonExpression, candidatesDict, positiveCore, negativeCore,
                                                   refinementLeft=refinementNum,
                                                   acceptanceCriterion=refinementCriterion,
                                                   balance=balance))
        return solutionExpressions

    def refine(self, skeletonExpression, candidatesDict, positiveCore, negativeCore,
               refinementLeft=0, acceptanceCriterion="weight>0.5",
               balance=True):
        solutionExpression = self.optimize_formula(skeletonExpression, candidatesDict, positiveCore, negativeCore,
                                                   balance=balance)
        empRate, satRate, weight = self.learn_independent_weight(solutionExpression)
        if refinementLeft > 0 and not criterion_satisfied(empRate, satRate, weight, acceptanceCriterion):
            print("# Refining since criterion not satisfied, {} tries left #".format(refinementLeft))
            newSkeleton = er.add_leaf_atom(skeletonExpression)
            refinementLeft -= 1
            return self.refine(newSkeleton, candidatesDict, positiveCore, negativeCore,
                               refinementLeft)
        else:
            print("# Solution is {} #".format(solutionExpression))
            return solutionExpression

    def optimize_formula(self, skeletonExpression, candidatesDict, positiveCore, negativeCore,
                         optimizationInstructions=["als2", "project", "als2", "project", "als2", "project"],
                         balance=True):
        exLearner = el.SampleBasedOptimizer(skeletonExpression, candidatesDict)

        exLearner.generate_fixedCores_sampleDf(self.sampleDf)
        exLearner.generate_target_and_filterCore_from_exampleCores(positiveCore, negativeCore)
        if balance:
            exLearner.balance_importance(positiveCore=positiveCore, negativeCore=negativeCore)
        exLearner.random_initialize_variableCoresDict()
        for optInstruction in optimizationInstructions:
            if optInstruction.startswith("als"):
                exLearner.als(int(optInstruction[3:]))
            if optInstruction == "project":
                exLearner.get_solution(adjustVariablesCoresDict=True)
        exLearner.get_solution()

        return exLearner.solutionExpression

    def add_independent_formula(self, expression, criterion=""):
        empRate, satRate, weight = self.learn_independent_weight(expression)
        if criterion_satisfied(empRate, satRate, weight, criterion):
            self.weightedFormulas.append([expression, weight])
            print("The expression {} with weight {} passed criterion {} and has been added.".format(expression, weight,
                                                                                                    criterion))
        else:
            print(
                "Expression {} with weight {} does not satisfy the criterion {}.".format(expression, weight, criterion))

    def learn_independent_weight(self, expression, verbose=False):
        return wees.calculate_weight(expression, self.sampleDf, verbose=verbose)

    def alternating_weight_optimization(self, sweepNum):
        estimator = wees.WeightEstimator([weightedFormula[0] for weightedFormula in self.weightedFormulas],
                                         sampleDf=self.sampleDf)
        estimator.alternating_optimization(sweepNum)
        self.weightedFormulas = [[estimator.formulaDict[formulaKey][0], estimator.formulaDict[formulaKey][3]] for
                                 formulaKey in estimator.formulaDict]

    def generate_mln(self):
        return mln.TensorMLN(expressionsDict=
                             {str(i): [self.weightedFormulas[i][0], self.weightedFormulas[i][1]]
                              for i in range(len(self.weightedFormulas))})


## Older name of the class
AtomicMLNLearner = SampleBasedMLNLearner


def criterion_satisfied(empRate, satRate, weight, criterion):
    criteria = criterion.split(",")
    for criterion in criteria:
        if criterion.startswith("weight"):
            threshold = float(criterion.split(">")[1])
            if weight < threshold:
                return False
        elif criterion.startswith("empRate"):
            threshold = float(criterion.split(">")[1])
            if empRate < threshold:
                return False
        elif criterion.startswith("satRate"):
            threshold = float(criterion.split("<")[1])
            if satRate > threshold:
                return False
        else:
            raise ValueError("Acceptance Criterion {} not understood.".format(criterion))
    return True
