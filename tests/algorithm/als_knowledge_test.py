from tnreason import encoding
from tnreason import engine

import numpy as np

from tnreason.algorithms import alternating_least_squares

networkCores = {
    **encoding.get_formulas_cores({"f1":["a1", "imp", "a2"]})
}

targetCores = {
    **als.copy_cores(encoding.get_formulas_cores({"f1":["a1", "imp", "a2"]}), "_tar", ["a1", "a2"]),
    "head": engine.get_core()(values=np.array([0, 1]), colors=["(a1_imp_a2)_tar"])
}



dataNum = 4
data = np.zeros(shape=(2,2,dataNum))
data[0,0,0] = 1
data[1,1,1] = 1
data[0,1,2] = 1
data[1,1,3] = 1
pos_phase=({"dataTensor" : engine.get_core()(values=data, colors=["a1", "a2", "dat"])},1)

neg_phase=({
    **encoding.get_formula_cores(["a1", "imp", "a2"]),
    "head": engine.get_core()(values=np.array([0, 1]), colors=["(a1_imp_a2)_tar"])
}, -1)

dataOptimizer = als.ALS(
    networkCores=networkCores,
    targetCores={"tarCore":engine.get_core()(values=np.ones(dataNum), colors=["dat"])},
    importanceList=
        [pos_phase, neg_phase]
        ,
    importanceColors=["dat"]
)

operator = dataOptimizer.compute_conOperator(["(a1_imp_a2)"], [2])
print(operator.values)

dataOptimizer.random_initialize(["estHead"], {"estHead": 2}, {"estHead": ["(a1_imp_a2)"]})


dataOptimizer.alternating_optimization(["estHead"], computeResiduum=False, sweepNum=1)
print(dataOptimizer.networkCores["estHead"].values)