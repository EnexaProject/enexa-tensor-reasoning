from tnreason import encoding
from tnreason import engine

import numpy as np

from tnreason.algorithms import gibbs_sampling

networkCores = {
    **encoding.get_formula_cores(["a1", "imp", "a2"])
}

dataNum = 4
data = np.zeros(shape=(2, 2, dataNum))
data[0, 0, 0] = 1
data[1, 1, 1] = 1
data[0, 1, 2] = 1
data[1, 0, 3] = 0.1 # Wrong datapoint -> give a small confidence



pos_phase = ({"dataTensor": engine.get_core()(values=data, colors=["a1", "a2", "dat"])}, 1)

neg_phase = ({
                 **encoding.get_formula_cores(["a1", "imp", "a2"]),
                 "head": engine.get_core()(values=np.array([0, 1]), colors=["(a1_imp_a2)_tar"])
             }, -1)

dataOptimizer = gibbs.Gibbs(
    networkCores=networkCores,
    importanceList=
    [pos_phase]# neg_phase]
    ,
    importanceColors=["dat"]
)

dataOptimizer.ones_initialization(["estHead"], {"estHead": 2}, {"estHead": ["(a1_imp_a2)"]})

dataOptimizer.alternating_sampling(["estHead"], computeResiduum=False, sweepNum=1, temperature=2)
print(dataOptimizer.networkCores["estHead"].values)
