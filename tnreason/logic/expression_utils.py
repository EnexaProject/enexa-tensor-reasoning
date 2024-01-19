import numpy as np


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


## Only used in Variable Learner, which has not been maintained!
def get_individuals(expression):
    if type(expression) == str:
        arguments = expression.split("(")[1][:-1]
        if "," in arguments:
            return arguments.split(",")
        else:
            return [arguments]
    elif expression[0] == "not":
        return get_individuals(expression[1])
    elif expression[1] == "and":
        return list(np.unique(np.concatenate((get_individuals(expression[0]), get_individuals(expression[2])))))
