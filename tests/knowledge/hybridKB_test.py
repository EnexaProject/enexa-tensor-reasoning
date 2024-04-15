from tnreason import knowledge

expressionsDict = {
    "e0": [["a1", "imp", "a2"], 2.123],
    "e1": [["a4", "eq", ["not", "a1"]], 2],
    "e2": [["a4", "xor", ["a5", "eq", "a1"]], 2],
    "e3": [["a6", "or", ["not", "a1"]], 2],
    "e4": ["a3", 100]
}
hybridKB = knowledge.HybridKnowledgeBase(weightedFormulasDict=expressionsDict,
                                         factsDict={"f2": ["not", "a2"],
                                                    "f3": "a2",
                                                    #"f3": "aasdfjaksödbfjklasghdfagsdfgajsfgahjsdgfjaasdfasdfasdfasdfasdfgsdf4",
                                                    "f4": "tev:TatbestandID_1000.0_SteuerinfoID_0.0(tid_sid)"},
                                         categoricalConstraintsDict={"c1": ["a1", "a2"]})

print(hybridKB.is_satisfiable())
exit()

hybridKB.create_cores()
hybridKB.to_yaml("./model.yaml")
print(hybridKB.ask("a1"))

hKB = knowledge.from_yaml("./model.yaml")

print(hKB.weightedFormulasDict)
print(hKB.factsDict)
print(hKB.categoricalConstraintsDict)

print(type(expressionsDict["e0"][1]))



hybridKB.include(
    knowledge.HybridKnowledgeBase(weightedFormulasDict={},
                                  factsDict={"f1": "a1"})
)
print(hybridKB.factsDict)
#                                         factsDict={"fact1":"a2", "fact2": ["not","a4"]})
# hybridKB.tell_constraint("a2")
print(hybridKB.ask_constraint("a2"))
print(hybridKB.annealed_map_query(variableList=["a3"], evidenceDict={"a1": 1, "a2": 1}))
print(hybridKB.ask([["not", "a2"], "or", "a1"]))
print(hybridKB.ask("aasdfjaksödbfjklasghdfagsdfgajsfgahjsdgfjaasdfasdfasdfasdfasdfgsdf4",
                   evidenceDict={"adsafasdfasdfaasdfasdfasdfasdfas3454537767899()sdfasdfasdf2": 0}))
print(hybridKB.ask("tev:TatbestandID_1000.0_SteuerinfoID_0.0(tid_sid)"))
print(hybridKB.exact_map_query(["a3"], evidenceDict={"a1": 1, "a2": 1}))

# hybridKB.visualize()
