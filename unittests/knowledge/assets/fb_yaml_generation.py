from tnreason import encoding

expressionsDict = {
    "e0": ["imp", "a1", "a2", 2.123],
    "e1": ["eq", "a4", ["not", "a1"], 2],
    "e2": ["xor", "a4", ["eq", "a5", "a1"], 2],
    "e3": ["or", "a6", ["not", "a1"], 2],
    "e4": ["a3", 100]
}

factsDict = {
    "f2": ["not", "a2"]
}

catConDict = {
    "c1": ["a1", "a3"]
}

encoding.save_as_yaml(modelSpec={
    "weightedFormulas": expressionsDict,
    "facts": factsDict,
    "categoricalConstraints": catConDict
}, savePath="./fb_backKb.yaml")

architecture = {
    "neur1": [["imp", "eq"],
              ["a1","a2"],
              ["a3", "a2"]
              ],
    "neur2": [["not", "id"],
              ["neur1", "a2"],
              ],
}

encoding.save_as_yaml(modelSpec=architecture, savePath="./fb_architecture.yaml")

boostSpecDict = {
    "method": "als",
    "sweeps": 5
}

encoding.save_as_yaml(modelSpec=boostSpecDict, savePath="./fb_als_boostSpec.yaml")

boostSpecDict = {
    "method": "gibbs",
    "annealingPattern": [
        [10,1],
        [10,10],
        [10,100]],
    "sweeps": 20
}

encoding.save_as_yaml(modelSpec=boostSpecDict, savePath="./fb_gibbs_boostSpec.yaml")