from tnreason.engine import polynomial_contractor as gsc

import numpy as np

core1 = gsc.PolynomialCore(np.array([[1, 2, 3], [0, 1, 0]]), ["a", "b"])
core2 = gsc.PolynomialCore(np.array([[1, 2, 3], [0, 1, 0]]), ["b", "c"])

print(core1.values.slices)

result = gsc.GenericSliceContractor({"c1": core1, "c2": core2}, ["a"]).contract()
print(result.values.slices)

result.add_identical_slices()
print(result.values.slices)