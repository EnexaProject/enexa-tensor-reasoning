from tnreason.engine import binary_slice_contractor as sc


import numpy as np


sc.BinarySliceCore(np.array([[1, 2], [1, 0]]), ["a", "b"])



values1 = [
    (1.1, {"a", "b"}, {"c"}),
    (0.9, set(), {"d"})
]
core1 = sc.BinarySliceCore(values1, ["a", "b", "c"])

values2 = [
    (1.1, {"b"}, {"a", "c"}),
    (2, {"a"}, set())
]
core2 = sc.BinarySliceCore(values2, ["a", "b", "c"])

contracted = sc.BinarySliceContractor(coreDict={
    "c1": core1,
    "c2": core2
}, openColors=["a", "b"]).contract()

print(contracted)