from tnreason.tensor import formula_tensors as ft
from tnreason.tensor import superposed_formula_tensors as sft

from tnreason.network import als
import pandas as pd


sampleDf = pd.read_csv("../assets/bbb_generated.csv")

skeleton = "p1"
candidatesDict = { "p1" : ["Unterschrank(z)","Moebel(z)","hatLeistungserbringer(x,y)"]}
sft = sft.SuperPosedFormulaTensor(skeleton, candidatesDict)

importanceCores = ft.DataTensor(sampleDf)

optimizer = als.ALS(
    sft.get_cores(),
    targetCores={}, # i.e. fitting the one tensor
    openTargetColors=list(sampleDf.columns)
)

optimizer.random_initialize(
    ["p1"], {"p1" : 3}, colorsDict={"p1" : ["p1"]}
)
optimizer.alternating_optimization(["p1"],2)
