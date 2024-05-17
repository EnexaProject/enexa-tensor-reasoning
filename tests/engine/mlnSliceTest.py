from tnreason import knowledge

hybridKB = knowledge.HybridKnowledgeBase(facts={
    "f1" : ["imp","a","b"]
})

from tnreason.engine import slice_contractor as sc

print(sc.SliceContractor(
    coreDict=hybridKB.create_cores(),
    openColors=["a","b"]
).contract())

core = sc.SliceContractor(
    coreDict=hybridKB.create_cores(),
    openColors=[]
).contract()

core.add_identical_slices()
print(core)
