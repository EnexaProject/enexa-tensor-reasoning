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


def get_symbols(expression):
    if isinstance(expression, str):
        return [expression]
    else:
        symbols = []
        for subexpression in expression:
            symbols = symbols + get_symbols(subexpression)
        return symbols

def replace_double_symbols(expression, replaceDict={}):
    if isinstance(expression, str):
        if expression in replaceDict:
            replaceDict[expression] = replaceDict[expression] + "_0"
        else:
            replaceDict[expression] = expression
        return expression, replaceDict
    elif len(expression) == 2:
        if expression[0] in replaceDict:
            replaceDict[expression[0]] = replaceDict[expression[0]] + "_0"
            expression[0] = replaceDict[expression[0]]
        else:
            replaceDict[expression[0]] = expression[0]
        rightReplaced, replaceDict = replace_double_symbols(expression[1], replaceDict)
        return [expression[0], rightReplaced], replaceDict
    elif len(expression) == 3:
        leftReplaced, replaceDict = replace_double_symbols(expression[0], replaceDict)
        if expression[1] in replaceDict:
            replaceDict[expression[1]] = replaceDict[expression[1]] + "_0"
            expression[1] = replaceDict[expression[1]]
        else:
            replaceDict[expression[1]] = expression[1]
        rightReplaced, replaceDict = replace_double_symbols(expression[2], replaceDict)
        return [leftReplaced, expression[1], rightReplaced], replaceDict


def decide_symbol_type(expression, symbol):
    if isinstance(expression, str):
        if expression == symbol:
            return "atom"
        else:
            return "unseen"
    elif len(expression) == 2:
        if expression[0] == symbol:
            return "unary"
        else:
            return decide_symbol_type(expression[1], symbol)
    elif len(expression) == 3:
        if expression[1] == symbol:
            return "binary"
        else:
            leftResult = decide_symbol_type(expression[0], symbol)
            rightResult = decide_symbol_type(expression[2], symbol)
            if rightResult == "unseen":
                return leftResult
            elif leftResult == "unseen":
                return rightResult
            else:
                if rightResult != leftResult:
                    raise ValueError("Symbol {} appears differently in expression {}.".format(symbol, expression))
                else:
                    return rightResult
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def get_subexpression(expression, connective):
    if isinstance(expression, str):
        return None
    elif len(expression) == 2:
        if expression[0] == connective:
            return expression
        return get_subexpression(expression[1], connective)
    elif len(expression) == 3:
        if expression[1] == connective:
            return expression
        else:
            left = get_subexpression(expression[0], connective)
            right = get_subexpression(expression[2], connective)
            if left is None:
                return right
            elif right is None:
                return left
            else:
                ValueError(
                    "Placeholder connective {} appears multiple in expression {}.".format(connective, expression))
