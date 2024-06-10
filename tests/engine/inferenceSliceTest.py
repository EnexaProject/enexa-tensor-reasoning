from tnreason import knowledge

hybridKB = knowledge.HybridKnowledgeBase(facts={
    "f1" : ["imp","a","b"]
})

provider = knowledge.InferenceProvider(hybridKB, contractionMethod="GenericSliceContractor")
result = provider.query(["a","b"])
result.add_identical_slices()

print(result.multiply(101))


print(result.sum_with(result))

