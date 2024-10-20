from pyqubo import Binary


def polynomialCore_to_pyqubo_hamiltonian(polynomialCore):
    binariesColorDict = {color: Binary(color) for color in polynomialCore.colors}
    hamiltonian = 0
    for val, positions in polynomialCore.values:
        hamiltonian = hamiltonian + create_potential(
            val, {key for key in positions if not positions[key]}, {key for key in positions if positions[key]},
            binariesColorDict
        )
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
    hamiltonian = polynomialCore_to_pyqubo_hamiltonian(polyCore)
    model = hamiltonian.compile()

    qubo, offset = model.to_qubo()
    print(qubo)
