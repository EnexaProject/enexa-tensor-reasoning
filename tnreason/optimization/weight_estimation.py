from optimization import satisfaction_counter as sc
from tnreason.logic import expression_calculus as ec

from tnreason.model import markov_logic_network as mln

import numpy as np


def calculate_satRate_bc(expression):
    return mln.calculate_dangling_basis(expression).count_satisfaction()

def solve_rate_equation(satRate, empRate):
    return -np.log(((1 - empRate) * satRate) / (empRate * (1 - satRate)))


def regularize_empRate(empRate, regFactor):
    return regFactor * (empRate - 0.5) + 0.5

def cutoff_weight(weight, cutoff):
    if weight > cutoff:
        return cutoff
    else:
        return  weight

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


def calculate_weight(expression, atomDict, filterCore=None, regFactor=1, verbose=False, check=True, cut = 20):
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

    print(solve_rate_equation(0.65,0.8))

    print(mln.calculate_dangling_basis(["not",["a","and",["not","b"]]]).values.shape)