from tnreason import knowledge

from tnreason.logic import coordinate_calculus as cc

import numpy as np

targetDict = {
    "t1": cc.CoordinateCore(np.random.binomial(10, 0.5, size=(2, 2)), ["a1", "b23"]),
    "t2": cc.CoordinateCore(np.random.binomial(20, 0.8, size=(2, 2, 5)), ["a1", "b23", "c4"])
}

hybridKB = knowledge.HybridInferer(
    {},  # "f1":[["a1","and",["not","a3"]], 10]},
    facts={"fact1": ["a1", "or", "a2"]},
    categoricalConstraints={
        "c1": ["a1", "a2", "a3"],
        "c2": ["a4"]
    }
)
print(hybridKB.facts.get_cores().keys())
print(hybridKB.formulaTensors.get_cores().keys())

optimizer = als.ALS(
    {**hybridKB.facts.get_cores(),
     **hybridKB.formulaTensors.get_cores()},
    targetCores=targetDict,
    importanceColors=["a1"]
)

optimizer.random_initialize(["con1", "a1_update"], {"con1": 3, "a1_update": 2},
                            {"con1": ["c1"], "a1_update": ["a1"]})

conoperator = optimizer.compute_conOperator(["a1"], [2])
print(conoperator.values)

print(optimizer.compute_residuum())
residua = optimizer.alternating_optimization(["con1", "a1_update"], computeResiduum=True)
print(residua)
print(optimizer.compute_residuum())