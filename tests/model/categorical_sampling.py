from tnreason.model import sampling

sampler = sampling.CategoricalGibbsSampler(
    {},#"f1":[["a1","and",["not","a3"]], 10]},
    factsDict={"fact1":["a1","or","a2"]},
    categoricalConstraintsDict={
        "c1" : ["a1","a2","a3"],
        "c2" : ["a4"]
    }
)

catEvidence = {"c1" : "a3",
               "c2" : 1}

#print(sampler.expressionsDict)
#print(sampler.factsDict)
#print(sampler.categoricalConstraintsDict)
#print(sampler.gibbs_step("c1", catEvidence))
#print(sampler.calculate_categorical_probability("c1"))

print(sampler.gibbs_sampling(2))