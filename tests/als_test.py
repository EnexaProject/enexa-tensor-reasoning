from tnreason.optimization import generalized_als as gals

from tnreason.logic import coordinate_calculus as cc, optimization_calculus as oc

import numpy as np

def check_core_equality(core0, core1):
    core0Norm = np.linalg.norm(core0.values)
    core0_negated = core0.negate(ignore_ones=True)
    difference = core0_negated.sum_with(core1)
    return np.linalg.norm(difference.values) < 0.001 * core0Norm

testFixedCoresDict = {
    "a": cc.CoordinateCore(np.random.binomial(n=1,p=0.5,size=(10, 7, 5)), ["l1", "y", "z"], "Sledz"),
    "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.5, size=(10, 5, 6)), ["l2", "z", "q"], "Jaszczur"),
    "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.5, size=(7, 4, 6)), ["y", "l3", "q"], "Sikorka"),
}

testVariablesCoresDict = {
    "a": cc.CoordinateCore(np.random.binomial(n=1,p=0.5,size=(10)), ["l1"], "Sledz"),
    "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.5, size=(10)), ["l2"], "Jaszczur"),
    "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.5, size=(4)), ["l3"], "Sikorka"),
}

expression = [["not","a"],"and",["b","and","c"]]

coresDictA = gals.create_core_dict(testVariablesCoresDict,testFixedCoresDict,"a")
coresDictB = gals.create_core_dict(testVariablesCoresDict,testFixedCoresDict,"b")
coresDictC = gals.create_core_dict(testVariablesCoresDict,testFixedCoresDict,"c")

operatorA, constantA = oc.calculate_operator_and_constant(coresDictA,
                                                          expression,
                                                          "a",
                                                          testVariablesCoresDict["a"].colors)
operatorB, constantB = oc.calculate_operator_and_constant(coresDictB,
                                                          expression,
                                                          "b",
                                                          testVariablesCoresDict["b"].colors)
operatorC, constantC = oc.calculate_operator_and_constant(coresDictC,
                                                          expression,
                                                          "c",
                                                          testVariablesCoresDict["c"].colors)

print(np.linalg.norm(operatorA.values),np.linalg.norm(constantA.values))
print(np.linalg.norm(operatorB.values),np.linalg.norm(constantB.values))
print(np.linalg.norm(operatorC.values),np.linalg.norm(constantC.values))

evaluationA = operatorA.contract_common_colors(testVariablesCoresDict["a"]).sum_with(constantA)
evaluationB = operatorB.contract_common_colors(testVariablesCoresDict["b"]).sum_with(constantB)
evaluationC = operatorC.contract_common_colors(testVariablesCoresDict["c"]).sum_with(constantC)

print(np.linalg.norm(evaluationA.values))
print(np.linalg.norm(evaluationB.values))
print(np.linalg.norm(evaluationC.values))

assert check_core_equality(evaluationA,evaluationB)
assert check_core_equality(evaluationA,evaluationC)
assert check_core_equality(evaluationB,evaluationC)