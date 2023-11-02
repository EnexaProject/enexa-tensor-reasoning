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

### OLD
def UNUSED_NEW_calculate_operator_and_constant(atom_dict, expression, variableKey, variableColors):
    if type(expression) == str:
        core = atom_dict[expression].clone()
        if affected_colors(core.colors, variableColors):
            return atom_dict[expression].clone(), atom_dict[expression].create_constant(variableColors, zero=True)
        else:
            return atom_dict[expression].create_constant([], zero=True), atom_dict[expression].clone()
    elif expression[0] == "not":
        operator, constant = calculate_operator_and_constant(atom_dict, expression[1], variableKey, variableColors)
        return operator.negate(ignore_ones=True), constant.negate()
    elif expression[1] == "and":
        operator0, constant0 = calculate_operator_and_constant(atom_dict, expression[0], variableKey, variableColors)
        operator2, constant2 = calculate_operator_and_constant(atom_dict, expression[2], variableKey, variableColors)

        toBeSummed = [constant0.compute_and(constant2),
                      constant0.compute_and(operator2),
                      operator0.compute_and(constant2),
                      operator0.compute_and(operator2)]

        operator = toBeSummed[3].create_constant([],zero=True)
        constant = toBeSummed[0].create_constant(variableColors,zero=True)
        for core in toBeSummed:
            if affected_colors(core.colors, variableColors):
                operator = operator.sum_with(core)
            else:
                constant = constant.sum_with(core)
        return operator, constant

def OLD_calculate_operator_and_constant(atom_dict,expression,variableKey,variableColors):
    if type(expression) == str:
        return atom_dict[expression].clone(), atom_dict[expression].create_constant(variableColors,zero=True)
    elif expression[0] == "not":
        if type(expression[1]) == str:
            if expression[1] == variableKey:
                constant = atom_dict[expression[1]].create_constant(variableColors)
                operator = atom_dict[expression[1]].clone().negate(ignore_ones=True)
            else:
                constant = atom_dict[expression[1]].create_constant(variableColors,zero=True)
                operator = atom_dict[expression[1]].clone().negate()
        else:
            operator, constant = calculate_operator_and_constant(atom_dict,expression[1],variableKey,variableColors)
            operator = operator.negate(ignore_ones = True)
            constant = constant.negate()

    elif expression[1] == "and":
        if type(expression[0]) == str:
            constant0 = atom_dict[expression[0]].create_constant(variableColors,zero=True)
            operator0 = atom_dict[expression[0]].clone()
        else:
            operator0, constant0 = calculate_operator_and_constant(atom_dict, expression[0],variableKey,variableColors)
        if type(expression[2]) == str:
            constant2 = atom_dict[expression[2]].create_constant(variableColors,zero=True)
            operator2 = atom_dict[expression[2]].clone()
        else:
            operator2, constant2 = calculate_operator_and_constant(atom_dict, expression[2],variableKey,variableColors)

        toBeSummed = [constant0.compute_and(constant2),
                      constant0.compute_and(operator2),
                      operator0.compute_and(constant2),
                      operator0.compute_and(operator2)]

        operator = toBeSummed[3].create_constant([],zero=True)
        constant = toBeSummed[0].create_constant(variableColors,zero=True)
        for core in toBeSummed:
            if variableColors[0] in core.colors:
                operator = operator.sum_with(core)
            else:
                constant = constant.sum_with(core)

        assert constant is not None, "Constant is None"
        assert operator is not None, "Operator is None"

    for opcolor in operator.colors:
        assert opcolor in constant.colors + variableColors, str(expression)
    for concolor in constant.colors:
        assert concolor in operator.colors + variableColors, str(expression)

    return operator, constant