from tnreason import knowledge

architectureDict = {
    "neur1": {
        "connectiveList": ["imp", "eq"],
        "candidatesList": [
            ["a1", "a2"],
            ["a3", "a4", "a2"]
        ]
    },
    "neur2": {
        "connectiveList": ["imp", "eq"],
        "candidatesList": [
            ["neur1"],
            ["a4", "a2"]
        ]
    }
}

specDict = {
    "method" : "als",
    "sweeps" : 10
}

kb = knowledge.from_yaml("./fun.yaml")

fBooster = knowledge.FormulaBooster(kb)
print(fBooster.find_candidate(architectureDict, specDict))