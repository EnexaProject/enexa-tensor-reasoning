from tnreason import knowledge
from tnreason import algorithms
from tnreason import encoding
from tnreason import engine

architectureDict =  {"neur1": [["imp"],
           ["a1"],
           ["a2", "a3"]]
 }
archCores = encoding.create_architecture(architectureDict)


from tnreason.algorithms import als
copied = als.copy_cores(archCores, "_out", ["a1","a2","a3"])

print(engine.contract({**archCores}, openColors=["neur1"]).values)

print(engine.contract({**archCores, **copied}, openColors=["neur1_p1_selVar","neur1_p1_selVar_out"]).values)

learner = algorithms.ALS(networkCores=
                         encoding.create_architecture(architectureDict),
                         importanceColors=["a1","a2","a3"])

print(learner.importanceColors)
conOperator = learner.compute_conOperator(
    ["neur1_p1_selVar"], [2]
)




print(conOperator.values)

exit()

genKB = knowledge.HybridKnowledgeBase(facts={
        "f1": ["a1"],
    },
        weightedFormulas={
            "wf1": ["imp", "a1", "a2", 1.1424],
            "wf1.5": ["a2", 0.2],
            "wf2": ["not", "a3", 0.2]
        }
    )
sampleDf = knowledge.InferenceProvider(genKB).draw_samples(10)

from tnreason.knowledge import distributions

empiricalDistribution = distributions.EmpiricalDistribution(sampleDf)







print(conOperator.values)