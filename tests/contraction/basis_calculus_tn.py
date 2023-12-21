from tnreason.contraction import bc_contraction_generation as bcg
from tnreason.contraction import core_contractor as coc

import numpy as np

## Optional
def inspect_formulaDict(insFormulaDict):
    for key in insFormulaDict:
        print("###")
        print(key)
        print(insFormulaDict[key].colors)
        print(insFormulaDict[key].values)

testDict = {
    "t1": [["P1", "and", "P2"], ["P1", "P2"], np.array([[0, 0], [0, 1]])],
    "t2": [[["not", "P1"], "and", ["not", "P2"]], ["P1", "P2"], np.array([[1, 0], [0, 0]])],
    "t3": [["not", ["P1", "and", "P2"]], ["P1", "P2"], np.array([[1, 1], [1, 0]])],
}

for testKey in testDict:
    print("## Test {} ###".format(testKey))
    expression, colors, values = testDict[testKey]
    print("Contraction of {} as basis calculus core.".format(expression))
    testFormulaDict = bcg.generate_factor_dict(expression, weight=1, headType="truthEvaluation")

    contractor = coc.CoreContractor(testFormulaDict, openColors=colors)
    contractor.optimize_coreList()
    contractor.create_instructionList_from_coreList()

    resultCore = contractor.contract()

    assert len(resultCore.colors) == 2
    assert np.linalg.norm(resultCore.values - values) == 0

for testKey in testDict:
    print("## Test {} ###".format(testKey))
    expression, colors, values = testDict[testKey]
    print("Contraction of {} as exponential factor.".format(expression))
    testFormulaDict = bcg.generate_factor_dict(expression, weight=1, headType="expFactor")

    contractor = coc.CoreContractor(testFormulaDict, openColors=colors)
    contractor.optimize_coreList()
    contractor.create_instructionList_from_coreList()

    resultCore = contractor.contract()

    assert len(resultCore.colors) == len(colors)
    assert np.linalg.norm(resultCore.values - np.exp(values)) == 0

for testKey in testDict:
    print("## Test {} ###".format(testKey))
    expression, colors, values = testDict[testKey]
    print("Contraction of {} as exponential factor.".format(expression))
    testFormulaDict = bcg.generate_factor_dict(expression, weight=1, headType="diffExpFactor")

    contractor = coc.CoreContractor(testFormulaDict, openColors=colors)
    contractor.optimize_coreList()
    contractor.create_instructionList_from_coreList()

    resultCore = contractor.contract()

    assert len(resultCore.colors) == len(colors)

    if len(values.shape) == 2:
        if values[0, 0] == 0:
            assert resultCore.values[0, 0] == 0
        else:
            assert np.linalg.norm(resultCore.values[0, 0] - np.exp(values[0, 0])) == 0
        if values[0, 1] == 0:
            assert resultCore.values[0, 1] == 0
        else:
            assert np.linalg.norm(resultCore.values[0, 1] - np.exp(values[0, 1])) == 0
        if values[1, 0] == 0:
            assert resultCore.values[1, 0] == 0
        else:
            assert np.linalg.norm(resultCore.values[1, 0] - np.exp(values[1, 0])) == 0
        if values[1, 1] == 0:
            assert resultCore.values[1, 1] == 0
        else:
            assert np.linalg.norm(resultCore.values[1, 1] - np.exp(values[1, 1])) == 0
