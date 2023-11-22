def calculate_core_dict(variableCoresDict, fixedCoresDict, legKey):
    core_dict = {}
    for coreKey in fixedCoresDict:
        if coreKey != legKey:
            core_dict[coreKey] = fixedCoresDict[coreKey].clone().contract_common_colors(
                variableCoresDict[coreKey].clone())
        else:
            core_dict[coreKey] = fixedCoresDict[coreKey].clone()
    return core_dict

def affected_colors(coreColors, variableColors):
    for color in coreColors:
        if color in variableColors:
            return True
    return False

def color_equivalence(core1,core2):
    for color in core1.colors:
        if color not in core2.colors:
            return False
    for color in core2.colors:
        if color not in core1.colors:
            return False
    return True

def sum_coreList(coreList):
    resultList = []
    while len(coreList)>0:
        core1 = coreList.pop(0)
        for core2 in coreList.copy():
            if color_equivalence(core1,core2):
                core1 = core1.sum_with(core2)
                coreList.pop(coreList.index(core2))
        resultList.append(core1)
    return resultList

def calculate_coreList(atom_dict,expression,variableColors):
    if type(expression) == str:
        return [atom_dict[expression]]
    elif expression[0] == "not":
        preCoreList = calculate_coreList(atom_dict, expression[1], variableColors)
        coreList = []
        for core in preCoreList:
            coreList.append(core.negate(ignore_ones = True))
        coreList.append(preCoreList[0].create_constant(variableColors))
        return sum_coreList(coreList)
    elif expression[1] == "and":
        toBeSummed = []
        preCoreList0 = calculate_coreList(atom_dict, expression[0], variableColors)
        preCoreList2 = calculate_coreList(atom_dict, expression[2], variableColors)
        for core0 in preCoreList0:
            for core2 in preCoreList2:
                toBeSummed.append(core0.compute_and(core2))
        return sum_coreList(toBeSummed)
    else:
        raise ValueError("Expression {} not understood.".format(expression))

def operator_constant_from_coreList(coreList, variableColors):
    assert len(coreList) <= 2, "CoreList too long!"
    if len(coreList) == 1:
        if affected_colors(coreList[0].colors,variableColors):
            return coreList[0], coreList[0].create_constant(variableColors, zero=True)
        else:
            raise ValueError("Variable Core is disconnected!")
    else:
        if affected_colors(coreList[0].colors,variableColors):
            return coreList[0], coreList[1]
        else:
            return coreList[1], coreList[0]

## To support the older tests
def calculate_operator_and_constant(atom_dict, expression, variableKey, variableColors):
    return operator_constant_from_coreList(calculate_coreList(atom_dict,expression,variableColors),variableColors)