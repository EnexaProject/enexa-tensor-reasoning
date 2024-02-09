def generate_list_from_rule(premises, head):
    expression = premises[0]
    for premise in premises[1:]:
        expression = [premise, "and", expression]
    expression = ["not", [["not", head], "and", expression]]

    return expression

def combine_with_same_connective(atoms, connective="and"):
    expression = atoms[0]
    for atom in atoms[1:]:
        expression = [atom, connective, expression]
    return expression

def generate_list_from_conjunctions(expression):
    ## ! Outputs the atoms in nested listed ! But used in real data tests.
    result = []
    for subExpression in expression:
        if isinstance(subExpression, list):
            result.extend(generate_list_from_conjunctions(subExpression))
        elif not (subExpression in ["and", "not"]):
            result.append(subExpression)
    return result


def generate_conjunctions(atoms):
    expression = atoms[0]
    for atom in atoms[1:]:
        expression = [atom, "and", expression]
    return expression


def generate_negated_conjunctions(positive_atoms, negated_atoms):
    return generate_conjunctions(positive_atoms + [["not", atom] for atom in negated_atoms])


def generate_from_generic_expression(expression):
    if type(expression) == str:
        return expression
    elif expression[0] == "not":
        return ["not", generate_from_generic_expression(expression[1])]
    elif expression[1] == "and":
        return [generate_from_generic_expression(expression[0]), "and", generate_from_generic_expression(expression[2])]
    elif expression[1] == "or":
        return ["not", [["not", generate_from_generic_expression(expression[0])], "and",
                        ["not", generate_from_generic_expression(expression[2])]]]
    elif expression[1] == "eq":
        left = generate_from_generic_expression(expression[0])
        right = generate_from_generic_expression(expression[2])
        return ["not", [
            ["not", [left, "and", right]]
            , "and",
            ["not", [["not", left], "and", ["not", right]]]
        ]]
    elif expression[1] == "imp":
        left = generate_from_generic_expression(expression[0])
        right = generate_from_generic_expression(expression[2])
        return ["not", [
            left, "and", ["not", right]
        ]]
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def replace_atoms(expression, atomDict):
    if isinstance(expression, str):
        if expression in atomDict:
            return atomDict[expression]
        else:
            return expression
    elif len(expression) == 2:
        if expression[0] in atomDict:
            connective = atomDict[expression[0]]
        else:
            connective = expression[0]
        return [connective, replace_atoms(expression[1], atomDict)]
    elif len(expression) == 3:
        if expression[1] in atomDict:
            connective = atomDict[expression[1]]
        else:
            connective = expression[1]
        return [replace_atoms(expression[0], atomDict), connective, replace_atoms(expression[2], atomDict)]
    else:
        raise ValueError("Expression {} not understood!".format(expression))

def decide_symbol_type(expression, symbol):
    if isinstance(expression, str):
        if expression==symbol:
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

def generate_pracmln_string(expression, weight):
    return str(weight) + " " + generate_pracmln_formulastring(expression)


def generate_pracmln_formulastring(expression):
    if type(expression) == str:
        return expression
    elif expression[0] == "not":
        return "!(" + generate_pracmln_string(expression[1]) + ")"
        # raise TypeError("pracmln string model does not yet support {}.".format(expression))
    elif expression[1] == "and":
        return generate_pracmln_string(expression[0]) + " ^ " + generate_pracmln_string(expression[2])


if __name__ == "__main__":
    and_expression = generate_from_generic_expression(["jaszczur", "or", ["sikorka", "or", ["not", "sledz"]]])

    print(generate_list_from_conjunctions(["a", "and", ["b", "and", "c"]]))
    print(and_expression)
