from tnreason import engine

def expressionsDict_to_PolynomialCore(expressionsDict):
    slices = expressionsDict_to_slices(expressionsDict)
    activeColors = []
    for (val, posDict) in slices:
        for color in posDict.keys():
            if color not in activeColors:
                activeColors.append(color)
    return engine.get_core("PolynomialCore")(values=slices, colors=activeColors)

def expressionsDict_to_slices(expressionsDict):
    slices = []
    for formulaName in expressionsDict:
        expression = expressionsDict[formulaName]
        if isinstance(expressionsDict[formulaName][-1], int) or isinstance(expressionsDict[formulaName][-1], float):
            slices = slices + [(val * expressionsDict[formulaName][-1], posDict) for (val, posDict) in
                               formula_to_slices(expression)]
        else:
            slices = slices + [(val, posDict) for (val, posDict) in formula_to_slices(expression)]
    return slices

def formula_to_slices(formula):
    if isinstance(formula, str):
        return [(1, {formula: 1})]
    elif formula[0] == "not":
        if isinstance(formula[1], str):
            return [(1, {formula[1]: 0})]
        else:
            return [(1, {})] + [(-value, posDict) for (value, posDict) in formula_to_slices(formula[1])]
    elif formula[0] == "and":
        slices = []
        for val1, posDict1 in formula_to_slices(formula[1]):
            for val2, posDict2 in formula_to_slices(formula[2]):
                if agreeing_dicts(posDict1, posDict2):
                    slices.append((val1 * val2, {**posDict1, **posDict2}))
        return slices
    else:
        raise ValueError("Formula {} not in not-and format!".format(formula))


def agreeing_dicts(pos1, pos2):
    for key in pos1:
        if key in pos2:
            if pos1[key] != pos2[key]:
                return False
    return True


if __name__ == "__main__":
    print(formula_to_slices(["not", "q"]))
    print(formula_to_slices(["not", ["and", "p", ["not", "q"]]]))

    core = expressionsDict_to_PolynomialCore(
        {"e1": ["not", ["and", "p", ["not", "q"]], 0.5627]}
    )
    print(core.values)