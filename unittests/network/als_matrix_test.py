import unittest

from tnreason import engine
from tnreason.network import als

import numpy as np

matrix = engine.get_core()(
    values = np.random.binomial(10, 0.5, size=(2, 2, 3)),
    colors = ["a1", "a2", "t"]
)

leftVector = engine.get_core()(
    values = np.random.binomial(10, 0.5, size=(2)),
    colors = ["a1"]
)

rightVector = engine.get_core()(
    values = np.random.binomial(10, 0.5, size=(2)),
    colors = ["a2"]
)

targetVector = engine.get_core()(
    values = np.random.binomial(10, 0.5, size=(3)),
    colors = ["t"]
)

optimizer = als.ALS(networkCores={"mat":matrix, "lvec":leftVector, "rvec":rightVector},
                    targetCores={"tar":targetVector},
                    openTargetColors=["t"])


class HybridKBTest(unittest.TestCase):

    def test_residuum_decay(self):
        residua = optimizer.alternating_optimization(["lvec","rvec"], computeResiduum=True)
        for i in range(residua.shape[0]):
            for j in range(residua.shape[1]-1):
                self.assertGreaterEqual(residua[i,j],residua[i,j+1])

    def test_decay_initialization(self):
        optimizer.random_initialize(["lvec","rvec"])
        residua = optimizer.alternating_optimization(["lvec","rvec"], computeResiduum=True)
        for i in range(residua.shape[0]):
            for j in range(residua.shape[1]-1):
                self.assertGreaterEqual(residua[i,j],residua[i,j+1],str(i)+"_"+str(j))

    def test_decay_new_initialization(self):
        optimizer.random_initialize(["new1","new2"], {"new1": 2, "new2": 2}, {"new1" : ["a1"], "new2": ["a2"]})
        residua = optimizer.alternating_optimization(["new1","new2"], computeResiduum=True)
        for i in range(residua.shape[0]):
            for j in range(residua.shape[1]-1):
                self.assertGreaterEqual(residua[i,j],residua[i,j+1],str(i)+"_"+str(j))