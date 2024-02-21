from tnreason.tensor import superposed_formula_tensors as sft

skeleton = ["p1","p2",["p4","p3"]]

candidatesDict = {
    "p1" : ["a", "b"],
    "p2" : ["and", "or"],
    "p4" : ["not"]
}
print(sft.SuperPosedFormulaTensor(skeleton, candidatesDict).get_cores())

