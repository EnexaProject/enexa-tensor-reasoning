import unittest

from tnreason import encoding
from tnreason import engine

import numpy as np

from tnreason.algorithms import alternating_least_squares as als

networkCores = {
    #**encoding.create_formulas_cores({"f1": ["imp", "a1", "a2"]})
    **encoding.create_raw_formula_cores(["imp", "a1", "a2"])
}

targetCores = {
    **als.copy_cores(encoding.create_formulas_cores({"f1": ["imp", "a1", "a2"]}), "_tar", ["a1", "a2"]),
    "head": engine.get_core()(values=np.array([0, 1]), colors=["(a1_imp_a2)_tar"])
}

optimizer = als.ALS(
    networkCores=networkCores,
    targetCores=targetCores,
    importanceColors=["a1", "a2"]
)


class AlsKnowledgeTest(unittest.TestCase):
    def test_operator_check(self):
        conOperator = optimizer.compute_conOperator(updateColors=["(imp_a1_a2)"])
        self.assertEqual(conOperator.values[1, 1], 3)
        self.assertEqual(conOperator.values[1, 0], 0)
        self.assertEqual(conOperator.values[0, 1], 0)
        self.assertEqual(conOperator.values[0, 0], 1)

        operator = engine.contract(coreDict={
            **networkCores,
            **als.copy_cores(networkCores, "_out", ["a1", "a2"])},
            openColors=["(imp_a1_a2)", "(imp_a1_a2)_out"])

        self.assertEqual(operator.values[0, 0], conOperator.values[0, 0])
        self.assertEqual(operator.values[0, 1], conOperator.values[0, 1])



    def test_world_recovery(self):
        optimizer.random_initialize(["estHead"], {"estHead": 2}, {"estHead": ["(imp_a1_a2)"]})
        optimizer.alternating_optimization(["estHead"], computeResiduum=False, sweepNum=1)

        self.assertEqual(optimizer.networkCores["estHead"].values[0], 0)
        self.assertEqual(optimizer.networkCores["estHead"].values[1], 1)

    def test_data_recovery(self):
        dataNum = 4
        data = np.zeros(shape=(2, 2, dataNum))
        data[0, 0, 0] = 1
        data[1, 1, 1] = 1
        data[0, 1, 2] = 1
        data[1, 1, 3] = 1

        dataOptimizer = als.ALS(
            networkCores=encoding.create_raw_formula_cores(["imp", "a1", "a2"]),
            targetCores={"tarCore": engine.get_core()(values=np.ones(dataNum), colors=["dat"])},
            importanceList=[({"dataTensor": engine.get_core()(values=data, colors=["a1", "a2", "dat"])}, 1)],
            importanceColors=["a1", "a2"]
        )

        dataOptimizer.random_initialize(["estHead"], {"estHead": 2}, {"estHead": ["(imp_a1_a2)"]})
        dataOptimizer.alternating_optimization(["estHead"], computeResiduum=False, sweepNum=1)
        self.assertEqual(dataOptimizer.networkCores["estHead"].values[0], 0)
        self.assertEqual(dataOptimizer.networkCores["estHead"].values[1], 1)
