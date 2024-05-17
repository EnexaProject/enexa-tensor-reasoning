from tnreason.engine import slice_contractor as sc


import numpy as np


sc.SliceCore(np.array([[1,2],[1,0]]), ["a","b"])



values1 = [
    (1.1, {"a", "b"}, {"c"}),
    (0.9, set(), {"d"})
]
core1 = sc.SliceCore(values1, ["a", "b", "c"])

values2 = [
    (1.1, {"b"}, {"a", "c"}),
    (2, {"a"}, set())
]
core2 = sc.SliceCore(values2, ["a", "b", "c"])

contracted = sc.SliceContractor(coreDict={
    "c1": core1,
    "c2": core2
}, openColors=["a", "b"]).contract()

print(contracted)