from tnreason import knowledge

hKB = knowledge.from_yaml("./model.yaml")
hKB.to_yaml("./model2.yaml")

expressionsDict =     {
        "e0" : [["a1","imp","a2"], 2],
        "e1" : [["a4","eq",["not","a1"]], 2],
        "e2":  [["a4", "xor", ["a5","eq", "a1"]], 2],
        "e3":  [["a6", "or", ["not", "a1"]], 2],
        "e4": ["a3", 100]
    }



hybridKB = knowledge.HybridKnowledgeBase(weightedFormulasDict=expressionsDict,
                                         factsDict={"f2":["not","a2"]})
hybridKB.include(
    knowledge.HybridKnowledgeBase(weightedFormulasDict={},
                                         factsDict={"f1":"a1"})
)
print(hybridKB.factsDict)
#                                         factsDict={"fact1":"a2", "fact2": ["not","a4"]})
#hybridKB.tell_constraint("a2")
print(hybridKB.ask_constraint("a2"))
print(hybridKB.annealed_map_query(variableList=["a3"],evidenceDict={"a1": 1, "a2":1}))
print(hybridKB.ask([["not","a2"],"or","a1"]))
print(hybridKB.exact_map_query(["a3"], evidenceDict={"a1":1,"a2":1}))

hybridKB.visualize()