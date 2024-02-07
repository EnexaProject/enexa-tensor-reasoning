from tnreason import  knowledge
hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulasDict={"f1": ["a1", 2]},
            factsDict={"constraint1": ["a1", "imp", "a2"]}
        )
print(hybridKB.annealed_map_query(["a3", "a4", "a1"]))

