from tnreason import knowledge

expressionsDict =     {
        "e0" : [["a1","imp","a2"], 2],
        "e1" : [["a4","eq",["not","a1"]], 2],
        "e2":  [["a4", "xor", ["a5","eq", "a1"]], 2],
        "e3":  [["a6", "or", ["not", "a1"]], 2],
    }

softKB = knowledge.SoftKnowledgeBase({})

print(softKB.ask(["a1","eq","a2"], evidenceDict={"a2":1}))

softKB.tell("a1",1)

print(softKB.ask(["a1","eq","a2"], evidenceDict={"a2":1}))