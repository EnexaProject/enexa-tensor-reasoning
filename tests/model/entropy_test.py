from tnreason.model import tensor_model as tm

from tnreason.model import entropies as ent

testDict = {
    "t0": ["a", 2],
    "t1": ["b", 1]
}

genDict = {
    "g0": [["a", "and", "b"], 2]
}

print(ent.expected_cross_entropy(testDict, genDict))

print(ent.expected_KL_divergence(testDict, genDict))
print(ent.expected_KL_divergence(genDict,genDict))