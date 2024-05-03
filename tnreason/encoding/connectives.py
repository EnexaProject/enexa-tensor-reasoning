import numpy as np

def get_unary_tensor(type):
    if type == "id":
        return np.eye(2)
    elif type == "not":
        return create_negation_tensor()


def get_binary_tensor(type):
    if type == "and":
        return create_conjunction_tensor()
    elif type == "or":
        return create_disjunction_tensor()
    elif type == "xor":
        return create_xor_tensor()
    elif type == "imp":
        return create_implication_tensor()
    elif type == "eq":
        return create_biconditional_tensor()
    else:
        raise ValueError("Binary connective {} not understood!".format(type))


def create_truth_vec():
    truthvec = np.zeros(2)
    truthvec[1] = 1
    return truthvec


def create_negation_tensor():
    negation_tensor = np.zeros((2, 2))
    negation_tensor[0, 1] = 1
    negation_tensor[1, 0] = 1
    return negation_tensor


def create_conjunction_tensor():
    and_tensor = np.zeros((2, 2, 2))
    and_tensor[0, 0, 0] = 1
    and_tensor[0, 1, 0] = 1
    and_tensor[1, 0, 0] = 1
    and_tensor[1, 1, 1] = 1
    return and_tensor


def create_disjunction_tensor():
    dis_tensor = np.zeros((2, 2, 2))
    dis_tensor[0, 0, 0] = 1
    dis_tensor[0, 1, 1] = 1
    dis_tensor[1, 0, 1] = 1
    dis_tensor[1, 1, 1] = 1
    return dis_tensor


def create_xor_tensor():
    xor_tensor = np.zeros((2, 2, 2))
    xor_tensor[0, 0, 0] = 1
    xor_tensor[0, 1, 1] = 1
    xor_tensor[1, 0, 1] = 1
    xor_tensor[1, 1, 0] = 1
    return xor_tensor


def create_implication_tensor():
    imp_tensor = np.zeros((2, 2, 2))
    imp_tensor[0, 0, 1] = 1
    imp_tensor[0, 1, 1] = 1
    imp_tensor[1, 0, 0] = 1
    imp_tensor[1, 1, 1] = 1
    return imp_tensor


def create_biconditional_tensor():
    bic_tensor = np.zeros((2, 2, 2))
    bic_tensor[0, 0, 1] = 1
    bic_tensor[0, 1, 0] = 1
    bic_tensor[1, 0, 0] = 1
    bic_tensor[1, 1, 1] = 1
    return bic_tensor










