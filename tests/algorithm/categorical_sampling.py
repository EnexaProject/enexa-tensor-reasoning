from tnreason import knowledge

hybridKB = knowledge.HybridKnowledgeBase(
    {},  # "f1":[["a1","and",["not","a3"]], 10]},
    factsDict={"fact1": ["a1", "or", "a2"]},
    categoricalConstraintsDict={
        "c1": ["a1", "a2", "a3"],
        "c2": ["a4"]
    }
)

from tnreason.algorithms import distributions

dist = distributions.TNDistribution(
    {**hybridKB.facts.get_cores(),
     **hybridKB.formulaTensors.get_cores()}
)

dist.gibbs_sampling(["c1", "a4"], {"c1": 3, "a4": 2}, sweepNum=10)


# print(sampler.expressionsDict)
# print(sampler.factsDict)
# print(sampler.categoricalConstraintsDict)
# print(sampler.gibbs_step("c1", catEvidence))
# print(sampler.calculate_categorical_probability("c1"))

# print(sampler.gibbs_sampling(2))
