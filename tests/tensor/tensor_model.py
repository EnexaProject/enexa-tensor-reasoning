from tnreason.tensor import tensor_model as tm

tRep = tm.TensorRepresentation(
    expressionsDict={"e1" : [["a","or","b"],1 ]},
    factsDict={"f1": "b"},
    categoricalConstraintsDict={"c1" : ["a","c"]}
)

coresDict = tRep.get_cores()
print(coresDict.keys())

from tnreason.contraction import contraction_visualization as cv

cv.draw_contractionDiagram(coresDict)