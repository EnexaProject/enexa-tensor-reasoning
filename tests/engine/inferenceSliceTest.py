from tnreason import knowledge

hybridKB = knowledge.HybridKnowledgeBase(facts={
    "f1" : ["imp","a","b"]
})

provider = knowledge.InferenceProvider(hybridKB, contractionMethod="SliceContractor")
result = provider.query(["a","b"])
result.add_identical_slices()

print(result)


from pyqubo import Binary

binariesDict = {color : Binary(color) for color in result.colors}

def create_potential(val, neg, pos):
    potential = 1
    for color in neg:
        potential = potential * (1- binariesDict[color])
    for color in pos:
        potential = potential * binariesDict[color]
    return val * potential

hamiltonian = 0
for val, neg, pos in result.values:
    hamiltonian = hamiltonian + create_potential(val , neg, pos)

print(hamiltonian)
model = hamiltonian.compile()

qubo, offset = model.to_qubo()
print(qubo)
