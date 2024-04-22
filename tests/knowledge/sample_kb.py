from tnreason import knowledge

testKb = knowledge.load_kb_from_yaml("./test_kb.yaml")

print(testKb.weightedFormulasDict)
print(testKb.gibbs_sample(variableList=testKb.atoms))
print(testKb.create_sampleDf(sampleNum=10))