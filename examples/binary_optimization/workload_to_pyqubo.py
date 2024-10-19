from pyqubo import Binary


def genericSliceCore_to_hamiltonian(genericSliceCore):
    binariesColorDict = {color: Binary(color) for color in genericSliceCore.colors}
    hamiltonian = 0
    for val, positions in genericSliceCore.values:
        hamiltonian = hamiltonian + create_potential(
            val, {key for key in positions if not positions[key]}, {key for key in positions if positions[key]},
            binariesColorDict
        )
    return hamiltonian

def binarySliceCore_to_hamiltonian(binarySliceCore):
    binariesColorDict = {color: Binary(color) for color in binarySliceCore.colors}
    hamiltonian = 0
    for val, neg, pos in binarySliceCore.values:
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
    from examples.cnf_representation import formula_to_polynomial_core as ftp

    polyCore = ftp.weightedFormulas_to_polynomialCore({
        "w1": ["imp", "a", "b", 0.678],
        "w2": ["a", 0.34]
    })
    hamiltonian = genericSliceCore_to_hamiltonian(polyCore)
    model = hamiltonian.compile()

    qubo, offset = model.to_qubo()
    print(qubo)
