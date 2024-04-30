from tnreason import knowledge

genKB = knowledge.HybridKnowledgeBase(facts={
    "f1" : ["a1"]
})

genKB.include(
    knowledge.HybridKnowledgeBase(
        weightedFormulas = {
            "wf1" : ["imp", "a1", "a2", 2.1424]
        }
    )
)
sampleDf = knowledge.HybridInferer(genKB).create_sampleDf(10)


learner = knowledge.HybridLearner(genKB)

boostSpecDict = {
    "method" : "als",
    "sweeps" : 10,
    "architecture":
        {"neur1" : [["imp","eq"],
                    ["a1","a2"],
                    ["a3","a4"]]
         },
    "acceptanceCriterion" : "always",
    "calibrationSweeps" : 10
}

learner.boost_formula(boostSpecDict, sampleDf)

calibrationSpecDict = {
    "calibrationSweeps" : 10
}

learner.calibrate_weights_on_data(calibrationSpecDict, sampleDf)