def generate_list_from_rule(premises, head):
    expression = premises[0]
    for premise in premises[1:]:
        expression = [premise, "and", expression]
    expression = ["not", [["not", head], "and", expression]]

    return expression


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


def remove_double_not(expression):
    if type(expression) == str:
        return expression
    elif expression[0] == "not":
        if expression[1][0] == "not":
            return remove_double_not(expression[1][1])
        else:
            return ["not", remove_double_not(expression[1])]
    elif expression[1] == "and":
        return [remove_double_not(expression[0]), "and", remove_double_not(expression[2])]
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def replace_atoms(expression, atomDict):
    if type(expression) == str:
        return atomDict[expression]
    else:
        if expression[0] == "not":
            return ["not", replace_atoms(expression[1], atomDict)]
        elif expression[1] == "and":
            return [replace_atoms(expression[0], atomDict), "and", replace_atoms(expression[2], atomDict)]


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
    assert remove_double_not(and_expression) == ['not',
                                                 [['not', 'jaszczur'], 'and', [['not', 'sikorka'], 'and', 'sledz']]], \
        "Generate from disjunctions or double not does not work"
    assert remove_double_not(["not", ["not", "sledz"]]) == "sledz", "Removing double not does not work"
