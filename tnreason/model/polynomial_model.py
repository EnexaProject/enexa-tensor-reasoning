class PolynomialRepresentation:
    def __init__(self, weightedFormulasDict):
        self.polynomialsDict = {key: Polynomial(weightedFormula=weightedFormulasDict[key]) for key in
                                weightedFormulasDict}
        self.monomialsDict = self.get_monomial_formulas()

    def get_monomial_formulas(self):
        formulasDict = {}
        for key in self.polynomialsDict:
            formulasDict = {**formulasDict,
                            **self.polynomialsDict[key].to_monomial_formulas()}
        return formulasDict

    def infer(self, evidenceDict):
        inferedMonomials = {}
        for key in self.monomialsDict:
            monomialVariables, weight = self.monomialsDict[key]
            for evidenceAtom in evidenceDict:
                if evidenceAtom in monomialVariables:
                    if evidenceDict[evidenceAtom]:
                        monomialVariables.pop(monomialVariables.index(evidenceAtom))
                    else:
                        monomialVariables = []
            if len(monomialVariables) > 0:
                inferedMonomials[key] = monomialVariables, weight
        return inferedMonomials


class Polynomial:
    def __init__(self, monomialsDict={}, weightedFormula=None):
        self.monomialsDict = monomialsDict
        if weightedFormula is not None:
            self.include_formula(weightedFormula)

    def include_formula(self, weightedFormula, key=None):
        if key is None:
            key = "f" + str(len(self.monomialsDict))
        self.monomialsDict = {**self.monomialsDict,
                              **formula_to_polynomial(weightedFormula[0], weightedFormula[1], key)}

    def to_monomial_formulas(self):
        return {key: [self.monomialsDict[key][0], self.monomialsDict[key][1]] for key in
                self.monomialsDict}


def formula_to_polynomial(formula, weight, key):
    if isinstance(formula, str):
        return {key: [[formula], weight]}
    elif formula[0] == "not":
        monomialsDict = formula_to_polynomial(formula[1], -1 * weight, key + "s")  # s for straight
        monomialsDict[key + "n"] = [[], 1]  # "n for negation
        return monomialsDict
    elif formula[1] == "and":
        rightDict = formula_to_polynomial(formula[0], 1, key + "r")
        leftDict = formula_to_polynomial(formula[2], 1, key + "l")
        monomialsDict = {}
        for rKey in rightDict:
            for lKey in leftDict:
                if len(rightDict[rKey]) > 0:
                    rVariables = set(rightDict[rKey][0])
                else:
                    rVariables = set()
                if len(leftDict[lKey]) > 0:
                    lVariables = set(leftDict[lKey][0])
                else:
                    lVariables = set()
                monomialsDict[rKey + lKey] = [list(rVariables | lVariables),
                                              weight * rightDict[rKey][1] * leftDict[lKey][1]]
        return monomialsDict
    else:
        raise ValueError("Formula {} not understood for polynomial transformation.".format(formula))


def monomial_to_formula(monomial):
    formula = monomial[0]
    for atom in monomial[1:]:
        formula = [formula, "and", atom]
    return formula


if __name__ == "__main__":
    poly = Polynomial(weightedFormula=[["a3", "and", ["not", ["a1", "and", "a2"]]], 10])
    print(poly.to_monomial_formulas())

    poly = Polynomial({
        "m1": [["a1"], 1],
        "m2": [["a2", "a3"], -1]
    })

    polyRep = PolynomialRepresentation({
        "fun": [["a3", "and", ["not", ["a1", "and", "a2"]]], 10]
    })
    print(polyRep.get_monomial_formulas())

    print(polyRep.infer(evidenceDict={"a1": 1, "a2": 0}))
