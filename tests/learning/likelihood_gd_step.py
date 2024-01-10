from tnreason.learning import alternating_mle as amle
from tnreason.model import generate_test_data as gtd
from tnreason.logic import coordinate_calculus as cc

from tnreason.contraction import contraction_visualization as cv
from tnreason.contraction import bc_contraction_generation as bcg

import numpy as np

atomDict = {
    "a1": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(2, 1, 3)), ["l1", "y", "z"], name="a"),
    "a2": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(1, 2, 3)), ["l2", "q", "z"], name="b"),
    "a3": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(3, 2, 3)), ["l3", "q", "z"], name="c"),
}

skeletonExpression = ["P1", "and", "P2"]
candidatesDict = {"P1": list(atomDict.keys()),
                  "P2": list(atomDict.keys()),
                  }

variableCoresDict = {
    "P1_variableCore": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P1", "H1"]),
    "P2_variableCore": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P2", "H1"]),
    "hiddenCore": cc.CoordinateCore(np.zeros(shape=(2)), ["H1"])
}

learnedFormulaDict = {
    "f0": ["a2", 1.2],
    "f1": [["a1", "and", "a2"], 1.3],
    "f2": ["a3", 2]
}



sampleDf = gtd.generate_sampleDf(learnedFormulaDict, 10)

optimizer = amle.GradientDescentMLE(skeletonExpression, candidatesDict, variableCoresDict, learnedFormulaDict, sampleDf=sampleDf)

#import tnreason.contraction.expression_evaluation as ee
#eval = ee.ExpressionEvaluator("a2")
#print(eval.evaluate_on_sampleDf(sampleDf).values)

optimizer.random_initialize_variableCoresDict()
#optimizer.create_atom_selectors()
optimizer.create_exponentiated_variables()

# cv.draw_contractionDiagram({"varExp":optimizer.variablesExpFactor,**optimizer.atomSelectorDict})
# cv.draw_contractionDiagram({**optimizer.variableCoresDict,**optimizer.fixedCoresDict})

print("Likelihood",optimizer.compute_likelihood())
print(optimizer.contract_partition())
print(optimizer.formulaCoreDict)

exit()

coreDict = bcg.generate_formulaCoreDict(learnedFormulaDict)
print(coreDict.keys())
for key in coreDict:
    print(key)
    print(coreDict[key].values)
    print(coreDict[key].colors)
exit()

fDict = bcg.generate_factor_dict(["a1","and","a2"])
print(list(fDict.keys()))
for key in coreDict:
    print(key)
    print(fDict[key].values)
exit()