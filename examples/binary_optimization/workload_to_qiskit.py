from qiskit.quantum_info import SparsePauliOp


def polynomialCore_to_qiskit_hamiltonian(polynomialCore):
    variableList = polynomialCore.colors  # Gives the interpretation of the qubits (stored only by their number)
    weightedMonomials = factorize_slices(polynomialCore.values)

    startVal, startPositions = weightedMonomials[0]
    hamiltonian = create_potential(startVal, startPositions, variableList)
    for val, positions in weightedMonomials[1:]:
        hamiltonian = hamiltonian + create_potential(val, positions, variableList)
    return hamiltonian


def create_potential(val, pos, variableList):
    """
    Shall represent each monomial as a Hamiltonian to be summed -> Unclear whether this is doing that!
    """
    listString = ""
    for variable in variableList:
        if variable in pos:
            listString = listString + "X"
        else:
            listString = listString + "I"
    return val * SparsePauliOp.from_list([(listString, 1.0)])


def factorize_slices(slices):
    weightedMonomials = []
    for factor, positionDict in slices:
        weightedMonomials = weightedMonomials + [(factor * weight, facList)
                                                 for weight, facList in factorize_product(positionDict)]
    return weightedMonomials


def factorize_product(positionDict):
    if len(positionDict) == 0:
        return [(1, [])]
    var, value = positionDict.popitem()
    sub_polynomial = factorize_product(positionDict)
    if value == 1:
        return [(weight, facList + [var]) for weight, facList in sub_polynomial]
    elif value == 0:
        return sub_polynomial + [(-1 * weight, facList + [var]) for weight, facList in sub_polynomial]
    else:
        raise ValueError("Value {} not supported in binary interpretation!".format(value))


if __name__ == "__main__":
    expression_dict = {"a": 0, "b": 1, "c": 1, "d": 0}
    result = factorize_product(expression_dict)
    print(result)

    from examples.cnf_representation import formula_to_polynomial_core as ftp

    polyCore = ftp.weightedFormulas_to_polynomialCore({
        "w1": ["imp", "a", "b", 0.678],
        "w2": ["a", 0.34]
    })
    hamiltonian = polynomialCore_to_qiskit_hamiltonian(polyCore)
    print(hamiltonian)
