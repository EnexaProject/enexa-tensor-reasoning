def get_all_variables(expressionList):
    variables = []
    for expression in expressionList:
        variables = variables + get_variables(expression)
    return variables


def get_variables(expression):
    if type(expression) == str:
        return [expression]
    elif expression[0] == "not":
        return get_variables(expression[1])
    elif expression[1] == "and":
        if type(expression[0]) == str:
            left_variables = [expression[0]]
        else:
            left_variables = get_variables(expression[0])
        if type(expression[2]) == str:
            right_variables = [expression[2]]
        else:
            right_variables = get_variables(expression[2])
        return left_variables + right_variables
    else:
        raise ValueError("Expression {} not understood.".format(expression))