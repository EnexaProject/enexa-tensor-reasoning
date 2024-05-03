from tnreason import knowledge

architectureDict = {
    "neur1": [
        ["imp", "eq"],
        ["a1", "a2"],
        ["a3", "a4", "a2"]
    ],
    "neur2": [
        ["imp", "eq"],
        ["neur1"],
        ["a4", "a2"],
    ]
}

specDict = {
    "method": "gibbs",
    "sweeps": 10
}

kb = knowledge.HybridInferer(weightedFormulas={
    "f1": [
        "eq",
        ["or", "a", "b"],
        "c",
        1.1
    ],
    "f2": [
        "imp",
        "a",
        ["or", "c", "d"],
        2.01
    ]
})

sampleDf = kb.create_sampleDf(10)

fBooster = knowledge.FormulaBooster(kb)
print(fBooster.find_candidate(architectureDict, specDict, sampleDf))
