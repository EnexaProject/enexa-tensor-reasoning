from tnreason import engine

import numpy as np


def create_emptyCoresDict(variableList, coreType="NumpyTensorCore", varDimDict=None):
    if varDimDict is None:
        varDimDict = {variableKey: 2 for variableKey in variableList}
    return {variableKey + "_trivialCore": engine.get_core(coreType=coreType)(np.ones(varDimDict[variableKey]),
                                                                             [variableKey],
                                                                             variableKey + "_trivialCore")
            for variableKey in variableList}


def get_all_variables(expressionList):
    variables = []
    for expression in expressionList:
        variables = variables + get_variables(expression)
    return np.unique(variables)


def get_variables(expression):
    if isinstance(expression, str):
        return [expression]
    elif len(expression) == 2:  # First entry is connective, second the subexpression
        return get_variables(expression[1])
    elif len(expression) == 3:  # Second entry is connective, first and last the subexpression
        if isinstance(expression[0], str):
            left_variables = [expression[0]]
        else:
            left_variables = get_variables(expression[0])
        if isinstance(expression[2], str):
            right_variables = [expression[2]]
        else:
            right_variables = get_variables(expression[2])
        return left_variables + right_variables
    else:
        raise ValueError("Expression {} not understood.".format(expression))