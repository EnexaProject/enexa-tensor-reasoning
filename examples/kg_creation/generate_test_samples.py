from tnreason import knowledge

dist = knowledge.HybridKnowledgeBase(weightedFormulas={
    "f1": ["or", "alpha", "beta", 0.4],
    "f2": ["gamma", 0.3]
})
samples = knowledge.InferenceProvider(dist).draw_samples(10)
samples["sample_index"] = samples.index
samples.to_csv("/Users/alexgoessmann/Documents/ENEXA/tnreason/version1/examples/kg_creation/example.csv")

