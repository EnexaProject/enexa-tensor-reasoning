from examples.cnf_representation import cnf_building as cb
from examples.cnf_representation import as_polynomial_core as ap

def formula_to_polynomialCore(expression):
    core = ap.clauseList_to_pcore(cb.cnf_to_dict(cb.to_cnf(expression)))
    core.add_identical_slices()
    return core

if __name__ == "__main__":
    testFormula = ["xor", ["eq", "a", "b"], ["not", ["imp", "b", "c"]]]
    core = formula_to_polynomialCore(testFormula)
    print(core)