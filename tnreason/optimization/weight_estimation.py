from tnreason.logic import satisfaction_counter as sc
from tnreason.logic import expression_calculus as ec

import numpy as np


def solve_rate_equation(satRate, empRate):
    return -np.log(((1 - empRate) * satRate) / (empRate * (1 - satRate)))


def regularize_empRate(empRate, regFactor):
    return regFactor * (empRate - 0.5) + 0.5


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


def calculate_weight(expression, atomDict, filterCore=None, regFactor=1):
    satRate = calculate_satRate(expression)
    empRate = regularize_empRate(calculate_empRate(expression, atomDict, filterCore), regFactor)
    return solve_rate_equation(satRate, empRate)


def partition_function(satRate, w):
    return (satRate * np.exp(w)) / (1 - satRate + satRate * np.exp(w))


if __name__ == "__main__":
    from tnreason.logic import coordinate_calculus as cc

    atomDict = {
        "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l1", "y", "z"], name="a"),
        "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l2", "q", "z"], name="b"),
        "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l3", "q", "z"], name="c"),
    }

    expression = ["b", "and", ["a", "and", ["not", "c"]]]
    filterCore = cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 10, 10, 7, 7, 5)),
                                   ["l1", "l2", "l3", "y", "q", "z"], name="c")

    checkEmpRate = calculate_empRate(expression, atomDict, filterCore)
    checkSatRate = calculate_satRate(expression)
    result = calculate_weight(expression, atomDict, filterCore)

    assert np.abs(partition_function(checkSatRate, result) - checkEmpRate) < 0.001, "Weight Calculation failed."
