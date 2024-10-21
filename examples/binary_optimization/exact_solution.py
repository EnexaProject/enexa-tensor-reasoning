import matplotlib.pyplot as plt
import dimod

def dimod_exact_solution(model):

    qubo, offset = model.to_qubo()

    exactsolver = dimod.ExactSolver()

    sampleset = exactsolver.sample_qubo(qubo)

    decoded_samples = model.decode_sampleset(sampleset)
    print(decoded_samples)
    best_sample = min(decoded_samples, key=lambda x: x.energy)

    energies = [sample.energy for sample in decoded_samples]

    return best_sample, energies

if __name__ == "__main__":
    from examples.cnf_representation import formula_to_polynomial_core as ftp
    from examples.binary_optimization import workload_to_pyqubo as wtp

    polyCore = ftp.weightedFormulas_to_polynomialCore({
        "w1": ["imp", "a", "b", 0.678],
        "w2": ["a", 0.34]
    })
    hamiltonian = wtp.polynomialCore_to_pyqubo_hamiltonian(polyCore)
    model = hamiltonian.compile()

    b_sample, energies = dimod_exact_solution(model)

    plt.scatter(range(len(energies)), energies)
    plt.show()

    print(b_sample.sample)
    print("Energy: {}".format(b_sample.energy))