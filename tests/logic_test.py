from tnreason.logic import expression_calculus as ec, basis_calculus as bc, coordinate_calculus as cc, \
    optimization_calculus as oc

import numpy as np


def check_array_equivalence(ar1, ar2):
    distance = np.linalg.norm(ar1.astype(float) - ar2.astype(float))**2
    return distance == 0

def check_core_equivalence(core1, core2):
    core1.reorder_colors(core2.colors)
    return check_array_equivalence(core1.values, core2.values)

def random_basis():
    vector = np.zeros((2))
    if np.random.binomial(n=1, p=0.5) == 1:
        vector[0] = 1
    else:
        vector[1] = 1
    return vector


def calculate_random_basis_core(shape):
    shapeProduct = np.prod(shape)
    core = np.zeros([2, shapeProduct])
    for i in range(shapeProduct):
        core[:, i] = random_basis()
    return core.reshape([2] + shape)


## CoordinateCalculus

core0 = cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(100, 10, 5)), ["a", "b", "c"], "Sledz")
core0_negated = core0.negate()
assert check_array_equivalence(core0_negated.values, np.ones((100, 10, 5)) - core0.values), "CC Negation does not work"

core1 = cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 100, 5)), ["b", "a", "c"], "Sledz")
summed = core0.sum_with(core1)
core0.reorder_colors(core1.colors)
assert check_array_equivalence(summed.values, core1.values + core0.values)

## BasisCalculus

bcore0 = bc.BasisCore(random_basis(), ["head"])
bcore1 = bc.BasisCore(random_basis(), ["head"])

bcore_negated = bcore0.negate()
assert np.dot(bcore0.values, bcore_negated.values) == 0, "BC Negation did not work"
assert np.linalg.norm(bcore_negated.values) == 1, "BC Negation did not work"

summed = bcore0.compute_and(bcore1)
assert bcore0.values[1] * bcore1.values[1] == summed.values[1], "BC Sum did not work"
assert np.linalg.norm(summed.values) == 1, "BC Sum did not work"

truthsummed = summed.calculate_truth()
assert summed.values[1] == truthsummed.values, "TruthCalculation did not work"

## Expression Calculus

basisDict = {
    "a": bc.BasisCore(calculate_random_basis_core([2]), ["head", "x"], name= "a"),
    "b": bc.BasisCore(calculate_random_basis_core([2]), ["head", "y"], name= "b")
}

coordinateDict = {
    "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "y", "z"], name = "a"),
    "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "q", "z"], name = "b"),
}

expression = ["a", "and", "b"]
basisExpression = ec.calculate_core(basisDict, expression)
assert basisExpression.values.shape == (2, 2, 2)
assert basisExpression.name == expression

coordinateExpression = ec.calculate_core(coordinateDict, expression)
assert len(coordinateExpression.colors) == 4
assert coordinateExpression.name == expression


expressionWithNegation = [["not","a"],"and",["not","b"]]
variableColors = ["x"]
variableKey = "a"
operator, constant = oc.calculate_operator_and_constant(coordinateDict,
                                                        expressionWithNegation,
                                                        variableKey,
                                                        variableColors)
assert not variableKey in constant.colors, "EC Constant Colors contain variablekey!"


expression = ["a","and",[["not","b"],"and",["not","c"]]]
#expression = [["a","and","b"], "and", "c"]
expression = ["not",["a","and",[["not","b"],"and",["not","c"]]]]
#expression = ["not",[["a","and",["not","b"]], "and", "c"]]
#expression = ["c","and",["b","and",["not",["not","a"]]]]

fixedCoresDict = {
    "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l1", "y", "z"], name = "a"),
    "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l2", "q", "z"], name = "b"),
    "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l3", "q", "z"], name = "c"),
}

variablesCoresDict = {
    "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10)), ["l1"], name = "a"),
    "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10)), ["l2"], name = "b"),
    "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10)), ["l3"], name = "c"),
}

coresDictA = oc.calculate_core_dict(variablesCoresDict,fixedCoresDict,"a")
coresDictB = oc.calculate_core_dict(variablesCoresDict,fixedCoresDict,"b")
coresDictC = oc.calculate_core_dict(variablesCoresDict,fixedCoresDict,"c")

operatorA, constantA = oc.calculate_operator_and_constant(coresDictA, expression, "a", ["l1"])
operatorB, constantB = oc.calculate_operator_and_constant(coresDictB, expression, "b", ["l2"])
operatorC, constantC = oc.calculate_operator_and_constant(coresDictC, expression, "c", ["l3"])

predA = operatorA.contract_common_colors(variablesCoresDict["a"]).sum_with(constantA)
predB = operatorB.contract_common_colors(variablesCoresDict["b"]).sum_with(constantB)
predC = operatorC.contract_common_colors(variablesCoresDict["c"]).sum_with(constantC)

assert check_core_equivalence(predA, predB), "EC Error in Operator Constant Calculation!"
assert check_core_equivalence(predB, predC), "EC Error in Operator Constant Calculation!"
assert check_core_equivalence(predA, predC), "EC Error in Operator Constant Calculation!"

computed_Or = variablesCoresDict["a"].compute_or(variablesCoresDict["a"])
assert check_core_equivalence(variablesCoresDict["a"],computed_Or), "CC Error in Or Calculus"

computed_Or = variablesCoresDict["a"].compute_or(variablesCoresDict["b"])
assert len(np.argwhere(computed_Or.values>1)) == 0, "CC Error in Or Calculus"