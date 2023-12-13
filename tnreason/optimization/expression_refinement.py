from tnreason.logic import expression_utils as eu

import numpy as np


def refine_expression(expression, mode, newAtom):
    if mode == "not":
        return ["not", expression]
    if mode == "and":
        return [newAtom, "and", expression]


def add_leaf_atom(expression):
    atoms = eu.get_variables(expression)
    new_atom = "P" + str(len(atoms) + 1)
    return random_modification(expression, new_atom)


def random_modification(expression, atom_name="NewAtom"):
    if type(expression) == str:
        if np.random.rand() < 0.5:
            return refine_expression(expression, "not", atom_name)
        else:
            return refine_expression(expression, "and", atom_name)
    elif expression[0] == "not":
        return ["not", random_modification(expression[1], atom_name)]
    elif expression[1] == "and":
        if np.random.rand() < 0.5:
            left = random_modification(expression[0], atom_name)
            right = expression[2]
        else:
            left = expression[0]
            right = random_modification(expression[2], atom_name)
        return [left, "and", right]


if __name__ == "__main__":
    print(add_leaf_atom([["not", "sledz"], "and", "sikorka"]))
