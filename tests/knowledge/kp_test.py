from tnreason import knowledge

testKB = knowledge.load_kb_from_yaml("./test_kb.yaml")
testKB.backCores = {"a" : 1}
print(testKB)
#testKB.visualize(evidenceDict={"(not_a1)":1})



exit()

cores = testKB.create_cores({"a1":0}, propagationReduction=True)

print('(imp_a1_a2)_headCore' in cores)
print(cores)


print(testKB.create_cores().keys())

evaluator = knowledge.KnowledgePropagator(testKB, evidenceDict={
    "a1" : 0
})
print(evaluator.evaluate())
