from tnreason.logic import coordinate_calculus as cc

import numpy as np

coreDict = {
    "c1": cc.CoordinateCore(np.random.binomial(10, 0.5, size=(3, 2)), ["a", "b"]),
    "c2": cc.CoordinateCore(np.random.binomial(20, 0.8, size=(3, 2, 5)), ["a", "b", "c"])
}

from tnreason.algorithms import distributions as dis

dist = dis.TNDistribution(coreDict)
assignment = dist.gibbs_sampling(["a", "b"], {"a": 3, "b": 2})
print(assignment)