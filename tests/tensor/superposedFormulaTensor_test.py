from tnreason.tensor import superposed_formula_tensors as sft

skeleton = ["p1","p2",["not","p3"]]

candidatesDict = {
    "p1" : ["a", "b"],
    "p2" : ["and", "or"]
}
print(sft.SuperPosedFormulaTensor(skeleton, candidatesDict).get_cores())

