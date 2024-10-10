import numpy as np
from tnreason import engine

def generate_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType="NumpyTensorCore"):
    if coreType == "NumpyTensorCore":
        values = np.zeros(inshape + outshape)
        for i in np.ndindex(*inshape):
            values[i + tuple([int(entry) for entry in function(*i)])] = 1 # Watch out: Numpy interpretes True as full slice!

    elif coreType == "PolynomialCore":
        values = []
        colors = incolors + outcolors
        for i in np.ndindex(*inshape):
            values.append((1, {colors[k]: assignment for k, assignment in
                               enumerate(i + tuple([int(entry) for entry in function(*i)]))}))
    else:
        raise ValueError("Core type not understood.")
    return engine.get_core(coreType=coreType)(values=values, colors=incolors + outcolors)


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


if __name__ == "__main__":
    ## Sanity check
    from tnreason import encoding
    for connectiveKey in ["imp","and","or","xor","eq"]:
        core = generate_relational_encoding([2, 2], [2], ["a", "b"], ["c"], get_connective_lambda(connectiveKey),
                                            coreType="NumpyTensorCore")
        print(np.linalg.norm(core.values - encoding.create_raw_formula_cores([connectiveKey, "a", "b"])["("+connectiveKey+"_a_b)_conCore"].values))
