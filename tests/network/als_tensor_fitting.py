from tnreason.tensor import formula_tensors as ft
from tnreason.tensor import superposed_formula_tensors as sft

from tnreason.network import als
import pandas as pd

from tnreason.tensor import tensor_model as tm

backgroundModel = tm.TensorRepresentation({"f": ["Moebel(z)", 2]})

sampleDf = pd.read_csv("../assets/bbb_generated.csv")[["Unterschrank(z)", "Moebel(z)","hatLeistungserbringer(x,y)"]]
dataTensor = ft.DataTensor(sampleDf)

skeleton = ["p1", "and", "p2"]
candidatesDict = {"p1": ["hatLeistungserbringer(x,y)", "Moebel(z)"],
                  "p2": ["Unterschrank(z)"]}
sft = sft.SuperPosedFormulaTensor(skeleton, candidatesDict)

importanceCores = ft.DataTensor(sampleDf)

optimizer = als.ALS(
    sft.get_cores(),
    targetCores={},  # i.e. fitting the one tensor
    openTargetColors=list(sampleDf.columns)
)

optimizer.random_initialize(
    ["p1_parCore", "p2_parCore"], {"p1_parCore": 2, "p2_parCore": 1},
    colorsDict={"p1_parCore": ["p1"], "p2_parCore": ["p2"]}
)
optimizer.alternating_optimization(["p1_parCore", "p2_parCore"], 1, importanceList = [(dataTensor.get_cores(), 1 / dataTensor.dataNum), (
    backgroundModel.get_cores(), -1 / backgroundModel.contract_partition())])

resCores = {
"p1_parCore": optimizer.networkCores["p1_parCore"],
"p2_parCore": optimizer.networkCores["p2_parCore"]
}
print(resCores)

print(optimizer.networkCores["p1_parCore"].values)
