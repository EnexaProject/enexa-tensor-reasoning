from tnreason import engine

import numpy as np


def create_emptyCoresDict(variableList, coreType="NumpyTensorCore", varDimDict=None, suffix="_trivialCore"):
    if varDimDict is None:
        varDimDict = {variableKey: 2 for variableKey in variableList}
    return {variableKey + suffix: engine.get_core(coreType=coreType)(np.ones(varDimDict[variableKey]),
                                                                             [variableKey],
                                                                             variableKey + suffix)
            for variableKey in variableList}


def get_all_variables(expressionsDict):
    variables = set()
    for expressionKey in expressionsDict:
        variables = variables | get_variables(expressionsDict[expressionKey])
    return list(variables)


def get_variables(expression):
    if isinstance(expression, str):
        return {expression}
    elif isinstance(expression, float) or isinstance(expression, int):
        return set()
    elif len(expression) == 1:
        return set(expression)
    else:
        atoms = set()
        for subExpression in expression[1:]:
            atoms = atoms | get_variables(subExpression)
        return atoms
