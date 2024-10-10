import numpy as np
from examples.rencoding.generate_rencoding import generate_relational_encoding


def get_connective_lambda(connectiveKey):
    if connectiveKey == "imp":
        return lambda a, b: [int(not a or b)]
    elif connectiveKey == "and":
        return lambda a, b: [int(a and b)]
    elif connectiveKey == "or":
        return lambda a, b: [int(a or b)]
    elif connectiveKey == "xor":
        return lambda a, b: [int(a ^ b)]
    elif connectiveKey == "eq":
        return lambda a, b: [int(a == b)]
    elif connectiveKey == "id":
        return lambda a: [int(a)]
    elif connectiveKey == "not":
        return lambda a: [int(not a)]


if __name__ == "__main__":
    ## Sanity check
    from tnreason import encoding

    for connectiveKey in ["imp", "and", "or", "xor", "eq"]:
        core = generate_relational_encoding([2, 2], [2], ["a", "b"], ["c"], get_connective_lambda(connectiveKey),
                                            coreType="NumpyTensorCore")
        print(np.linalg.norm(core.values - encoding.create_raw_formula_cores([connectiveKey, "a", "b"])[
            "(" + connectiveKey + "_a_b)_conCore"].values))

    for connectiveKey in ["id","not"]:
        core =generate_relational_encoding([2], [2], ["a"], ["c"], get_connective_lambda(connectiveKey),
                                            coreType="NumpyTensorCore")
        print(np.linalg.norm(core.values - encoding.create_raw_formula_cores([connectiveKey, "a"])[
            "(" + connectiveKey + "_a)_conCore"].values))
