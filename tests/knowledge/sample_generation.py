from tnreason import knowledge

format = {
    "weightedFormulas" : {
        "f1": ["a", 2],
        "f3": ["and","b",["not","d"],100]
    }
}

kb = knowledge.HybridKnowledgeBase(
    **format
)

from tnreason.encoding import auxiliary_cores
#print(auxiliary.get_all_atoms({key: format["weightedFormulas"][key][:-1]
#                             for key in format["weightedFormulas"]}))

#exit()
print(kb.weightedFormulas,"formulas")
print(kb.atoms)
kb.create_cores()



infprovider = knowledge.InferenceProvider(kb)
print(infprovider.distribution.atoms)
print(infprovider.draw_samples(10))


from tnreason import encoding

#encoding.get_variables({
#["a",2]
#})