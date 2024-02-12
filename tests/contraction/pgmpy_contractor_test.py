from tnreason import contraction

from tnreason.logic import coordinate_calculus as cc
import numpy as np

contracted = contraction.get_contractor("PgmpyVariableEliminator")(
coreDict = {
    "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "y", "z"], name="a"),
    "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["x", "y2", "z"], name="a2")
},
    openColors = ["y2","x"]
).contract()

print(contracted.values)
print(contracted.colors)