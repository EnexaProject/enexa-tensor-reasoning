from tnreason import knowledge

genKB = knowledge.HybridKnowledgeBase(facts={
    "f1": ["a1"]
})

genKB.include(
    knowledge.HybridKnowledgeBase(
        weightedFormulas={
            "wf1": ["imp", "a1", "a2", 1.1424],
            "wf1.5": ["a2", 0.2],
            "wf2": ["not", "a3", 50.2]
        }
    )
)

print(genKB)
exit()

sampleDf = knowledge.InferenceProvider(genKB).draw_samples(100)

learner = knowledge.HybridLearner(knowledge.HybridKnowledgeBase(
    weightedFormulas={"w1": ["not", "a3", 2],
                      "w2": ["a2", -1]}
))

boostSpecDict = {
    "method": "als",
    "sweeps": 10,
    "headNeurons": ["neur1"],
    "architecture":
        {"neur1": [["imp"],
                   ["a1"],
                   ["a3", "a2"]]
         },
    "acceptanceCriterion": "always",
    "calibrationSweeps": 2
}

learner.graft_formula(boostSpecDict, sampleDf)

exit()
calibrationSpecDict = {
    "calibrationSweeps": 2
}

learner.calibrate_weights_on_data(calibrationSpecDict, sampleDf)
