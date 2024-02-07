from tnreason import  knowledge
#hybridKB = knowledge.HybridKnowledgeBase(
#            weightedFormulasDict={"f1": ["a1", 2]},
#            factsDict={"constraint1": ["a1", "imp", "a2"]}
#        )
#print(hybridKB.annealed_map_query(["a3", "a4", "a1"]))


sampleRepetition = 100
hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulasDict={"f1": [["a1", "imp", "a2"], 10]},
            factsDict={"f2": "a4"},
            categoricalConstraintsDict={"c1": ["a1", "a2", "a3"]}
        )

for rep in range(sampleRepetition):
    sample = hybridKB.annealed_map_query(["a1", "a2", "a3"])