from tnreason import knowledge

architectureDict = {
    "neur1": {
        "connectiveList": ["imp", "eq"],
        "candidatesList": [
            ["a1", "a2"],
            ["a3", "a4", "a2"]
        ]
    }
}

specDict = {
    "method" : "gibbs",
    "sweeps" : 10
}

kb = knowledge.from_yaml("./fun.yaml")

fBooster = knowledge.FormulaBooster(kb)
fBooster.find_candidate(architectureDict, specDict)