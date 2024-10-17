import numpy as np
from tnreason.engine import workload_to_tentris as wt

testCore1 = wt.HypertrieCore(np.array([[0, 2], [0.12, -1.1]]), colors=["a", "b"])
#print([entry for entry in testCore1.values])
print(testCore1.to_NumpyTensorCore().values)


testCore2 = wt.HypertrieCore(np.array([[1.1, 2], [0.12, -1.1]]), colors=["a", "c"])

testContractor = wt.HypertrieContractor({"1": testCore1, "2": testCore2}, ["b", "c"])
print(testContractor.einsum().to_NumpyTensorCore().values)