from examples.cnf_representation import cnf_building as cb
from examples.cnf_representation import as_polynomial_core as ap


def formula_to_polynomialCore(expression):
    """
    Turns an expression into a Polynomial Core based on the contraction of its clauses
    """
    core = ap.clauseList_to_pcore(simplify_clauseList(cnf_to_dict(cb.to_cnf(expression, uppushAnd=False))))
    core.add_identical_slices()
    return core


def cnf_to_dict(expression, atomsOnly=True):
    # atomsOnly: Whether all keys in the dictionary refer to 2-dimensional categorical varibables (standard)
    if isinstance(expression, str):
        return [{expression: 1}]  ## Then a positive Literal
    elif len(expression) == 2:
        if expression[0] == "not":  ## Then a negative Literal
            return [{expression[1]: 0}]
        elif expression[0] == "id":
            return cnf_to_dict(expression[1])
    elif len(expression) == 3:
        if expression[0] == "and":
            return [entry for entry in cnf_to_dict(expression[1]) if len(entry) != 0] + [entry for entry in
                                                                                         cnf_to_dict(expression[2]) if
                                                                                         len(entry) != 0]
        elif expression[0] == "or":
            combinedClauses = []
            for leftClause in cnf_to_dict(expression[1]):
                for rightClause in cnf_to_dict(expression[2]):
                    if atomsOnly and all([rightClause[key] == leftClause[key] for key in
                                          set(rightClause.keys()) & set(leftClause.keys())]):
                        combinedClauses.append({**leftClause, **rightClause})
            if len(combinedClauses) == 0:  ## If all clauses got trivial
                return [dict()]
            return combinedClauses


def simplify_clauseList(clauseList):
    simplifiedList = clauseList.copy()
    for clause in clauseList:
        simplifiedList.remove(clause)
        if not any([all([clause[key] == testClause[key] for key in set(clause.keys()) & set(testClause.keys())])
                    and set(testClause.keys()) <= clause.keys() for testClause in simplifiedList]):
            # Check whether any testClause in simplified list is contained in the clause
            simplifiedList.append(clause)
    return simplifiedList


if __name__ == "__main__":
    testFormula = ["xor", ["eq", "a", "b"], ["not", ["imp", "b", "c"]]]
    formula_to_polynomialCore(testFormula)

    testFormula = ["eq", ["eq", "a", "b"], ["not", ["imp", "b", "c"]]]
    formula_to_polynomialCore(testFormula)

    testFormula = ["or", ["and", "b", "c"], ["and", "b", "c"]]
    clauseList = cnf_to_dict(cb.to_cnf(testFormula, uppushAnd=False))
    assert len(simplify_clauseList(clauseList)) == 2 # Needs to be {"b": 1}, {"c": 1}
    formula_to_polynomialCore(testFormula)
