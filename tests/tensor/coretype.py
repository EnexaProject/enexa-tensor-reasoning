from tnreason import tensor

import numpy as np

core = tensor.get_core("NumpyTensorCore")(
    np.array([1,2]),
    ["a"]
)

from tnreason.tensor import model_cores as mcore

mcore.create_subExpressionCores(["a","and","b"], "blub")

mcore.create_conCore(["not","a"])

tensor.get_core()(mcore.create_negation_tensor(), ["a","b"], "blub")

from tnreason.tensor import formula_tensors as ft

formulaTensor = ft.FormulaTensor(["not","a"])
print(formulaTensor.get_cores())
formulaTensor.infer_on_evidenceDict({"a":1})