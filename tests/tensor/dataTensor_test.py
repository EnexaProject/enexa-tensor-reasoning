from tnreason.tensor import formula_tensors as ft

import pandas as pd


print(ft.CategoricalConstraint(["a","b","c"], name="cat").get_cores())

sampleDf = pd.read_csv("../assets/bbb_generated.csv")

dataTensor = ft.DataTensor(sampleDf)

print(dataTensor.compute_shannon_entropy())






