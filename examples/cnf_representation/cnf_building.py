def to_cnf(expression, uppushAnd=False):  ## Allowing for ors before ands if uppushAnd=False
    if not isinstance(expression, str) and len(expression) == 1: # To handle stripped weightedFormulas
        expression = expression[0]
    expression = eliminate_eq_xor(expression)
    expression = eliminate_imp(expression)
    expression = groundpush_not(expression)
    if uppushAnd:
        expression = uppush_and(expression)
    return expression


def eliminate_eq_xor(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:
        return [expression[0], eliminate_eq_xor(expression[1])]
    elif len(expression) == 3:
        if expression[0] == "eq":
            return ["and", ["imp", eliminate_eq_xor(expression[1]), eliminate_eq_xor(expression[2])],
                    ["imp", eliminate_eq_xor(expression[2]), eliminate_eq_xor(expression[1])]]
        elif expression[0] == "xor":
            return ["not", ["and", ["imp", eliminate_eq_xor(expression[1]), eliminate_eq_xor(expression[2])],
                            ["imp", eliminate_eq_xor(expression[2]), eliminate_eq_xor(expression[1])]]]
        else:
            return [expression[0], eliminate_eq_xor(expression[1]), eliminate_eq_xor(expression[2])]
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def eliminate_imp(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:
        return [expression[0], eliminate_imp(expression[1])]
    elif len(expression) == 3:
        if expression[0] == "imp":
            return ["or", ["not", eliminate_imp(expression[1])], eliminate_imp(expression[2])]
        else:
            return [expression[0], eliminate_imp(expression[1]), eliminate_imp(expression[2])]
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def groundpush_not(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:  # Then assume that connective is not
        if isinstance(expression[1], str):  ## Already at literal level
            return expression
        elif len(expression[1]) == 2:
            return groundpush_not(expression[1][1])  # Case of double not
        elif len(expression[1]) == 3:
            if expression[1][0] == "and":
                return ["or", groundpush_not(["not", expression[1][1]]), groundpush_not(["not", expression[1][2]])]
            elif expression[1][0] == "or":
                return ["and", groundpush_not(["not", expression[1][1]]), groundpush_not(["not", expression[1][2]])]
            else:
                raise ValueError("Expression {} not groundpushable!".format(expression))
    elif len(expression) == 3:
        return [expression[0], groundpush_not(expression[1]), groundpush_not(expression[2])]

    else:
        raise ValueError("Expression {} not groundpushable!".format(expression))


def uppush_and(expression): ## Redundant, only when aiming at a CNF in nested language -> Better to go to clauseLists before that!
    while not and_above_or_checker(expression):
        expression = and_or_modify(expression)
    return expression


def and_or_modify(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:
        return [expression[0], and_or_modify(expression[1])]
    elif len(expression) == 3:
        if expression[0] == "or":
            if not_starts_with_and(expression[1]) and not not_starts_with_and(expression[2]):
                return ["and", ["or", expression[1], expression[2][1]], ["or", expression[1], expression[2][2]]]
            elif not not_starts_with_and(expression[1]) and not_starts_with_and(expression[2]):
                return ["and", ["or", expression[2], expression[1][1]], ["or", expression[2], expression[1][2]]]
            elif not not_starts_with_and(expression[1]) and not not_starts_with_and(expression[2]):
                return ["and",
                        ["and", ["or", expression[1][1], expression[2][1]], ["or", expression[1][1], expression[2][2]]],
                        ["and", ["or", expression[1][2], expression[2][1]], ["or", expression[1][2], expression[2][2]]]]
        return [expression[0], and_or_modify(expression[1]), and_or_modify(expression[2])]


def not_starts_with_and(expression):
    if isinstance(expression, str):
        return True
    else:
        return not expression[0] == "and"


def and_above_or_checker(expression):
    if isinstance(expression, str):
        return True
    elif len(expression) == 2:
        return True
    elif len(expression) == 3:
        if expression[0] == "or":
            return not_containing_and(expression)
        else:
            return and_above_or_checker(expression[1]) and and_above_or_checker(expression[2])


def not_containing_and(expression):
    if isinstance(expression, str):
        return True
    elif len(expression) == 2:
        return True
    elif len(expression) == 3:
        if expression[0] == "and":
            return False
        else:
            return not_containing_and(expression[1]) and not_containing_and(expression[2])


if __name__ == "__main__":
    testFormula = ["or", ["and", "b", "c"], ["and", "b", "c"]]
    cnf = to_cnf(testFormula, uppushAnd=False)
    print(cnf)
    cnf = to_cnf(testFormula, uppushAnd=True)
    print(cnf)

    testFormula = ["not", ["eq", "a", "b"]]
    cnf = to_cnf(testFormula)
    print(cnf)

    testFormula = ["eq", ["eq", "a", "b"], ["not", ["imp", "b", "c"]]]
    cnf = to_cnf(testFormula)
    print(cnf)
