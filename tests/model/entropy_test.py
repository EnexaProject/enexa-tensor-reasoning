from tnreason.model import entropies as ent
import tnreason.model.generate_test_data as gtd

testDict = {
    "t0": ["a", 2],
    "t1": ["b", 1]
}

genDict = {
    "g0": [["a", "and", "b"], 2]
}

sampleDf = gtd.generate_sampleDf(genDict, 100)
print(ent.empirical_shannon_entropy(sampleDf))

print(ent.expected_cross_entropy(testDict, genDict))
print(ent.expected_KL_divergence(testDict, genDict))
print(ent.expected_KL_divergence(genDict,genDict))