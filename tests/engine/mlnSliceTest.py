from tnreason import knowledge

hybridKB = knowledge.HybridKnowledgeBase(facts={
    "f1" : ["imp","a","b"]
})

from stubs import binary_slice_contractor as sc

print(sc.BinarySliceContractor(
    coreDict=hybridKB.create_cores(),
    openColors=["a","b"]
).contract())

core = sc.BinarySliceContractor(
    coreDict=hybridKB.create_cores(),
    openColors=[]
).contract()

core.add_identical_slices()
print(core)
