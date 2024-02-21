from tnreason.tensor import formula_tensors as ft

import pandas as pd


sampleDf = pd.read_csv("../assets/bbb_generated.csv")

ft.DataTensor(sampleDf)
