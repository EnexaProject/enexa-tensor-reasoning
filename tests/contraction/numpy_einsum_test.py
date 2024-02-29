from tnreason.tensor import formula_tensors as ft
from tnreason import contraction
import pandas as pd

cores1 = {**ft.FormulaTensor(["a", "and", ["not", "c_2"]]).get_cores(),
          **ft.FormulaTensor("b").get_cores()}
contractor = contraction.get_contractor("NumpyEinsum")(cores1, openColors=["a","b"])

cores = ft.FormulaTensor(["a", "and", "b"]).get_cores()
contractor = contraction.get_contractor("NumpyEinsum")(cores, openColors=["a"])
contractionResult = contractor.contract()
print(contractionResult.values)