from tnreason import engine

import numpy as np


emptyCoreSuffix = "_trivialCore"

def create_emptyCoresDict(variableList, coreType="NumpyTensorCore", varDimDict=None, suffix=emptyCoreSuffix):
    if varDimDict is None:
        varDimDict = {variableKey: 2 for variableKey in variableList}
    return {variableKey + suffix: engine.get_core(coreType=coreType)(np.ones(varDimDict[variableKey]),
                                                                     [variableKey], variableKey + suffix)
            for variableKey in variableList}


def get_all_atoms(expressionsDict):
    atoms = set()
    for key in expressionsDict:
        atoms = atoms | get_atoms(expressionsDict[key])
    return list(atoms)


def get_atoms(expression):
    if isinstance(expression, str):  ## Then an atom
        return {expression}
    elif len(expression) == 1:  ## Then an atomic formula
        return {expression[0]}
    else:  ## Then a formula with connective in first position
        atoms = set()
        for subExpression in expression[1:]:
            atoms = atoms | get_atoms(subExpression)
        return atoms
