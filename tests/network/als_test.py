from tnreason import knowledge

from tnreason.logic import coordinate_calculus as cc

import numpy as np

targetDict = {
    "t1": cc.CoordinateCore(np.random.binomial(10, 0.5, size=(2, 2)), ["a1", "b23"]),
    "t2": cc.CoordinateCore(np.random.binomial(20, 0.8, size=(2, 2, 5)), ["a1", "b23", "c4"])
}

hybridKB = knowledge.HybridKnowledgeBase(
    {},  # "f1":[["a1","and",["not","a3"]], 10]},
    factsDict={"fact1": ["a1", "or", "a2"]},
    categoricalConstraintsDict={
        "c1": ["a1", "a2", "a3"],
        "c2": ["a4"]
    }
)
print(hybridKB.facts.get_cores().keys())
print(hybridKB.formulaTensors.get_cores().keys())

from tnreason.network import als

optimizer = als.ALS(
    {**hybridKB.facts.get_cores(),
     **hybridKB.formulaTensors.get_cores()},
    targetCores=targetDict,
    openTargetColors=["a1"]
)
optimizer.random_initialize(["con1", "a1_update"], {"con1": 3, "a1_update": 2},
                            {"con1": ["c1"], "a1_update": ["a1"]})
optimizer.alternating_optimization(["con1", "a1_update"])
optimizer.optimize_core("con1")
