from tnreason.tensor import formula_tensors as ft
from tnreason.tensor import superposed_formula_tensors as sft

from tnreason.network import als
import pandas as pd

from tnreason.tensor import tensor_model as tm

backgroundModel = tm.TensorRepresentation({"f": ["Moebel(z)", 2]})

sampleDf = pd.read_csv("../assets/bbb_generated.csv")
dataTensor = ft.DataTensor(sampleDf)

skeleton = ["p1", "xor", "p2"]
candidatesDict = {"p1": ["Unterschrank(z)", "Moebel(z)", "hatLeistungserbringer(x,y)"],
                  "p2": ["Unterschrank(z)", "Moebel(z)", "hatLeistungserbringer(x,y)"]}
sft = sft.SuperPosedFormulaTensor(skeleton, candidatesDict)

importanceCores = ft.DataTensor(sampleDf)

optimizer = als.ALS(
    sft.get_cores(),
    targetCores={},  # i.e. fitting the one tensor
    openTargetColors=list(sampleDf.columns)
)

optimizer.random_initialize(
    ["p1", "p2"], {"p1": 3, "p2": 3}, colorsDict={"p1": ["p1"], "p2": ["p2"]}
)
optimizer.alternating_optimization(["p1", "p2"], 2, importanceList=[(dataTensor.get_cores(), 1 / dataTensor.dataNum), (
backgroundModel.get_cores(), -1 / backgroundModel.contract_partition())])
