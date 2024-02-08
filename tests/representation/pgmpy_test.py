from tnreason import knowledge
from tnreason.representation import pgmpy_inference as pinf

hybridKB = knowledge.from_yaml("../knowledge/fun.yaml")
inferer = pinf.from_hybridKB(hybridKB)
print(inferer.model.nodes)


result = inferer.query(["a5","a6"],{"a4":1})
result.normalize()
print(result.values)
print(type(result))
print(inferer.generate_sampleDf(10, 10, method="Gibbs"))


import tnreason.logic.coordinate_calculus as cc
import numpy as np

inferer = pinf.PgmpyInferer({"f2": cc.CoordinateCore(np.random.normal(size=(5, 2, 4)), ["a", "b", "c"]),
                            "f1": cc.CoordinateCore(np.random.normal(size=(5, 2)), ["a", "d"])})

# inferer.visualize()

print(inferer.map_query(["c"], {"a": 1}))
result = inferer.query(["c", "d"], {"b": 0})
print(result.values)
print(result.variables)
print(dir(result))