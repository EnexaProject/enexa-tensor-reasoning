from tnreason.optimization import weight_estimation as wees

from tnreason.contraction import  core_contractor as coc

def check_pos_neg_weight(estimator, tboFormulaKey):
    posCoreDict = {formulaKey: estimator.coreDict[formulaKey].weighted_exponentiation(estimator.formulaDict[formulaKey][3])
               for formulaKey in estimator.formulaDict if formulaKey != tboFormulaKey}
    negCoreDict = posCoreDict.copy()
    posCoreDict[tboFormulaKey] = estimator.coreDict[tboFormulaKey].clone()
    negCoreDict[tboFormulaKey] = estimator.coreDict[tboFormulaKey].negate()
    posContractor = coc.CoreContractor(posCoreDict)
    posContractor.optimize_coreList()
    positiveExpWeight = posContractor.contract().values

    assert len(positiveExpWeight.shape) == 0

    negContractor = coc.CoreContractor(negCoreDict)
    negContractor.optimize_coreList()
    negativeExpWeight = negContractor.contract().values

    assert len(negativeExpWeight.shape) == 0

    return negativeExpWeight, positiveExpWeight

formulaList = [
    "P1",
    "P2",
    ["not", ["P1","and","P2"]]
]

estimator = wees.WeightEstimator(formulaList)


estimator.calculate_independent_satRates()

#print(estimator.formulaDict)
print(estimator.alternating_optimization(sweepNum=10))
#print(estimator.formulaDict)



## Use for a TEST case!



