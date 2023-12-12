from tnreason.optimization import satisfaction_counter as sc

from tnreason.logic import expression_calculus as ec
from tnreason.logic import core_contractor as coc

import numpy as np


class WeightEstimator:
    def __init__(self, formulaList):
        self.formulaDict = {"f" + str(i): [formula, 0, 0, 0] for i, formula in enumerate(formulaList)}
        self.coreDict = {}  ## CoordinateCores of each formula -> Calculated in self.calculate_independent_satRates, to initialize the alternating optimization

    def calculate_independent_satRates(self):
        for formulaKey in self.formulaDict:
            basis_core = ec.calculate_expressionCore(self.formulaDict[formulaKey][0])
            self.formulaDict[formulaKey][1] = basis_core.count_satisfaction()
            self.coreDict[formulaKey] = basis_core

    def calculate_empRates(self, atomDict):
        for formulaKey in self.formulaDict:
            self.formulaDict[formulaKey][2] = calculate_empRate(self.formulaDict[formulaKey][0], atomDict)

    def independent_estimation(self, atomDict, calculateEmp=True, calculateSat=True, cut=100):
        if calculateEmp:
            self.calculate_empRates(atomDict)
        if calculateSat:
            self.calculate_independent_satRates()

        for formulaKey in self.formulaDict:
            self.formulaDict[formulaKey][3] = cutoff_weight(
                solve_rate_equation(self.formulaDict[formulaKey][1], self.formulaDict[formulaKey][2]), cut)

    def alternating_optimization(self, atomDict, sweepNum):
        weightTracker = np.empty((sweepNum + 1, len(self.formulaDict)))
        self.independent_estimation(atomDict)
        weightTracker[0] = [self.formulaDict[formulaKey][3] for formulaKey in self.formulaDict]
        for sweepPos in range(sweepNum):
            for i, formulaKey in enumerate(self.formulaDict):
                self.formula_optimization(formulaKey)
                weightTracker[sweepPos + 1, i] = self.formulaDict[formulaKey][3]
        return weightTracker

    def formula_optimization(self, tboFormulaKey):

        posCoreDict = {formulaKey: self.coreDict[formulaKey].weighted_exponentiation(self.formulaDict[formulaKey][3])
                       for formulaKey in self.formulaDict if formulaKey != tboFormulaKey}
        negCoreDict = posCoreDict.copy()
        posCoreDict[tboFormulaKey] = self.coreDict[tboFormulaKey].clone()
        negCoreDict[tboFormulaKey] = self.coreDict[tboFormulaKey].negate()
        posContractor = coc.CoreContractor(posCoreDict)
        posContractor.optimize_coreList()
        positiveExpWeight = posContractor.contract().values

        assert len(positiveExpWeight.shape) == 0

        negContractor = coc.CoreContractor(negCoreDict)
        negContractor.optimize_coreList()
        negativeExpWeight = negContractor.contract().values

        assert len(negativeExpWeight.shape) == 0

        negPosQuotient = negativeExpWeight / positiveExpWeight
        empRate = self.formulaDict[tboFormulaKey][2]
        self.formulaDict[tboFormulaKey][3] = np.log(negPosQuotient * (empRate / (1 - empRate)))


def calculate_satRate_bc(expression):
    return ec.calculate_expressionCore(expression).count_satisfaction()


def solve_rate_equation(satRate, empRate):
    return -np.log(((1 - empRate) * satRate) / (empRate * (1 - satRate)))


def regularize_empRate(empRate, regFactor):
    return regFactor * (empRate - 0.5) + 0.5


def cutoff_weight(weight, cutoff):
    if weight > cutoff:
        return cutoff
    else:
        return weight


def calculate_empRate(expression, atomDict, filterCore=None):
    expressionCore = ec.calculate_core(atomDict, expression)
    if filterCore is None:
        expressionResults = expressionCore.values.flatten()
        empSatNum = np.sum(expressionResults)
        dataNum = expressionResults.shape[0]
    else:
        expressionCore = expressionCore.compute_and(filterCore)
        empSatNum = np.sum(expressionCore.values)
        dataNum = np.sum(filterCore.values)
    return empSatNum / dataNum


def calculate_satRate(expression):
    variables = np.unique(ec.get_variables(expression))
    modelNum = 2 ** len(variables)
    satNum = sc.count_satisfaction(expression)
    return satNum / modelNum


def calculate_weight(expression, atomDict, filterCore=None, regFactor=1, verbose=False, check=True, cut=20):
    satRate = calculate_satRate(expression)

    if check:
        assert satRate == calculate_satRate_bc(expression), "Saturation Rate does not coincide with Basis Calculus!"

    empRate = regularize_empRate(calculate_empRate(expression, atomDict, filterCore), regFactor)
    weight = cutoff_weight(solve_rate_equation(satRate, empRate), cut)
    if verbose:
        print("## Calculationg the weight of {} ##".format(expression))
        print("World satisfaction rate: {}".format(satRate))
        print("Data satifcation rate: {}".format(empRate))
        print("Calculated weight: {}".format(weight))
    return empRate, satRate, weight


def partition_function(satRate, w):
    return (satRate * np.exp(w)) / (1 - satRate + satRate * np.exp(w))


if __name__ == "__main__":
    from tnreason.logic import coordinate_calculus as cc

    from matplotlib import pyplot as plt

    atomDict = {
        "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l1", "y", "z"], name="a"),
        "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l2", "q", "z"], name="b"),
        "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l3", "q", "z"], name="c"),
    }

    expressionList = [["b", "and", ["a", "and", ["not", "c"]]],
                      "a"]
    estimator = WeightEstimator(expressionList)
    estimator.independent_estimation(atomDict)
    weightTracker = estimator.alternating_optimization(atomDict, 10)

    plt.imshow(weightTracker, cmap="coolwarm")
    plt.colorbar()
    plt.show()

    ## Independent Calculation (Old)
    expression = ["b", "and", ["a", "and", ["not", "c"]]]

    filterCore = cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 10, 10, 7, 7, 5)),
                                   ["l1", "l2", "l3", "y", "q", "z"], name="c")

    checkEmpRate = calculate_empRate(expression, atomDict, filterCore)
    checkSatRate = calculate_satRate(expression)
    result = calculate_weight(expression, atomDict, filterCore)
