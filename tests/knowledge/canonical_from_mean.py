from tnreason import knowledge

kb = knowledge.EntropyMaximizer(
    expressionsDict= { "e1" : ["a"],
                       "e2" : ["b"]},
    satisfactionDict={"e1": 1, "e2":0.55}
).get_optimized_knowledge_base()

print(kb)