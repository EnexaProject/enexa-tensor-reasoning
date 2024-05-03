from tnreason import knowledge

genKB = knowledge.HybridKnowledgeBase(facts={
    "f1" : ["a1"]
})

genKB.include(
    knowledge.HybridKnowledgeBase(
        weightedFormulas = {
            "wf1" : ["imp", "a1", "a2", 1.1424],
            "wf1.5": ["a2", 10],
            "wf2" : ["not","a3", 10.2]
        }
    )
)
sampleDf = knowledge.InferenceProvider(genKB).draw_samples(100)


print(sampleDf)

learner = knowledge.HybridLearner(knowledge.HybridKnowledgeBase())

boostSpecDict = {
    "method" : "als",
    "sweeps" : 10,
    "architecture":
        {"neur1" : [["eq","imp"],
                    ["a1"],
                    ["a3","a2"]]
         },
    "acceptanceCriterion" : "always",
    "calibrationSweeps" : 2
}

learner.boost_formula(boostSpecDict, sampleDf)


exit()
calibrationSpecDict = {
    "calibrationSweeps" : 2
}

learner.calibrate_weights_on_data(calibrationSpecDict, sampleDf)