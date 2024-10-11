def get_connectives(connectiveKey):
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

def get_unary_connective_selector(connectiveList):
    return lambda l, a: get_connectives(connectiveList[l])(a)

def get_binary_connective_selector(connectiveList):
    return lambda l, a, b: get_connectives(connectiveList[l])(a, b)

def get_connective_selector(connectiveList):
    if connectiveList[0] in ["imp","and","or","xor","eq"]:
        return lambda l, a, b : get_connectives(connectiveList[l])(a,b)
    else:
        return lambda l, a: get_connectives(connectiveList[l])(a)