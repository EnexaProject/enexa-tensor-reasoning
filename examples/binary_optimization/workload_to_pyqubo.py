from pyqubo import Binary


def core_to_hamiltonian(sliceCore):
    binariesColorDict = {color: Binary(color) for color in sliceCore.colors}
    hamiltonian = 0
    for val, neg, pos in sliceCore.values:
        hamiltonian = hamiltonian + create_potential(val, neg, pos, binariesColorDict)
    return hamiltonian


def create_potential(val, neg, pos, binariesDict):
    potential = 1
    for color in neg:
        potential = potential * (1 - binariesDict[color])
    for color in pos:
        potential = potential * binariesDict[color]
    return val * potential


if __name__ == "__main__":
    from tnreason import knowledge

    hybridKB = knowledge.HybridKnowledgeBase(facts={
        "f1": ["imp", "a", "b"]
    })

    provider = knowledge.InferenceProvider(hybridKB, contractionMethod="SliceContractor")
    result = provider.query(["a", "b"])
    result.add_identical_slices()

    hamiltonian = core_to_hamiltonian(result)
    model = hamiltonian.compile()

    qubo, offset = model.to_qubo()
    print(qubo)
