import unittest

from tnreason import encoding
from tnreason import engine

import numpy as np

from tnreason.network import als

networkCores = {
    **encoding.get_formula_cores(["a1", "imp", "a2"])
}

targetCores = {
    **als.copy_cores(encoding.get_formula_cores(["a1", "imp", "a2"]), "_tar", ["a1", "a2"]),
    "head": engine.get_core()(values=np.array([0, 1]), colors=["(a1_imp_a2)_tar"])
}

optimizer = als.ALS(
    networkCores=networkCores,
    targetCores=targetCores,
    openTargetColors=["a1", "a2"]
)

dataNum = 4
data = np.zeros(shape=(2, 2, dataNum))
data[0, 0, 0] = 1
data[1, 1, 1] = 1
data[0, 1, 2] = 1
data[1, 1, 3] = 1

dataOptimizer = als.ALS(
    networkCores=networkCores,
    targetCores={"tarCore": engine.get_core()(values=np.ones(dataNum), colors=["dat"])},
    importanceList=[({"dataTensor": engine.get_core()(values=data, colors=["a1", "a2", "dat"])}, 1)],
    openTargetColors=["a1", "a2"]
)


class HybridKBTest(unittest.TestCase):
    def test_operator_check(self):
        operator = engine.contract(coreDict={
            **networkCores,
            **als.copy_cores(networkCores, "_out", ["a1", "a2"])},
            openColors=["(a1_imp_a2)", "(a1_imp_a2)_out"])

        conOperator = optimizer.compute_conOperator(updateColors=["(a1_imp_a2)"], updateShape=[2])

        self.assertEquals(operator.values[0, 0], conOperator.values[0, 0])
        self.assertEquals(operator.values[0, 1], conOperator.values[0, 1])

    def test_world_recovery(self):
        optimizer.random_initialize(["estHead"], {"estHead": 2}, {"estHead": ["(a1_imp_a2)"]})
        optimizer.alternating_optimization(["estHead"], computeResiduum=False, sweepNum=1)

        self.assertEquals(optimizer.networkCores["estHead"].values[0], 0)
        self.assertEquals(optimizer.networkCores["estHead"].values[1], 1)

    def test_data_recovery(self):
        dataOptimizer.random_initialize(["estHead"], {"estHead": 2}, {"estHead": ["(a1_imp_a2)"]})
        dataOptimizer.alternating_optimization(["estHead"], computeResiduum=False, sweepNum=1)
        self.assertEquals(dataOptimizer.networkCores["estHead"].values[0], 0)
        self.assertEquals(dataOptimizer.networkCores["estHead"].values[1], 1)
