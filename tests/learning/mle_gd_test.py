from tnreason.learning import alternating_mle as amle

from tnreason.logic import coordinate_calculus as cc

import numpy as np

atomDict = {
    "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l1", "y", "z"], name="a"),
    "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l2", "q", "z"], name="b"),
    "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l3", "q", "z"], name="c"),
}

skeletonExpression = ["P1", "and", "P2"]
candidatesDict = {"P1": list(atomDict.keys()),
                  "P2": list(atomDict.keys()),
                  }

variableCoresDict = {
    "v1": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P1", "H1"]),
    "v2": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P2", "H1"]),
    "hiddenCore": cc.CoordinateCore(np.zeros(shape=(2)), ["H1"])
}

learnedFormulaDict = {
    "f0": ["b", 10],
    "f1": [["not", ["a", "and", "b"]], 5],
    "f2": ["c", 2]
}

import tnreason.model.generate_test_data as gtd

sampleDf = gtd.generate_sampleDf(learnedFormulaDict, 100)

optimizer = amle.GradientDescentMLE(skeletonExpression, candidatesDict, variableCoresDict, learnedFormulaDict, sampleDf)

# optimizer.create_exponentiated_variables()

optimizer.random_initialize_variableCoresDict()
print(optimizer.alternating_gradient_descent(10, 1))

print(optimizer.compute_empirical_KL_divergence(sampleDf))