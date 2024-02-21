from tnreason import tensor

import numpy as np


def get_expression_string(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:
        assert isinstance(expression[0], str)
        return expression[0] + "_" + get_expression_string(expression[1])
    elif len(expression) == 3:
        assert isinstance(expression[1], str)
        return "(" + get_expression_string(expression[0]) + "_" + expression[1] + get_expression_string(
            expression[2]) + ")"


def create_conCore(expression, coreType="NumpyTensorCore"):
    expressionString = get_expression_string(expression)

    if isinstance(expression, str):
        return {expressionString + "_conCore": tensor.get_core(coreType=coreType)(np.ones(2), [expressionString],
                                                                                  expressionString + "_conCore")}
    elif len(expression) == 2:
        preExpressionString = get_expression_string(expression[1])
        return {expressionString + "_conCore": tensor.get_core(coreType=coreType)(
            get_unary_tensor(expression[0]),
            [preExpressionString, expressionString],
            expressionString + "_conCore")
        }

    elif len(expression) == 3:
        leftExpressionString = get_expression_string(expression[0])
        rightExpressionString = get_expression_string(expression[1])
        return {expressionString + "_conCore": tensor.get_core(coreType=coreType)(get_binary_tensor(expression[1]),
                                                                                  [leftExpressionString,
                                                                                   rightExpressionString,
                                                                                   expressionString],
                                                                                  expressionString + "_conCore")}
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def create_conCores(expression, coreType="NumpyTensorCore"):
    if isinstance(expression, str):
        return create_conCore(expression, coreType=coreType)
    elif len(expression) == 2:
        return {**create_conCore(expression, coreType=coreType),
                **create_conCores(expression[1], coreType=coreType)}
    elif len(expression) == 3:
        return {**create_conCore(expression, coreType=coreType),
                **create_conCores(expression[0], coreType=coreType),
                **create_conCores(expression[2], coreType=coreType)
                }


def create_headCore(expression, headType, weight=None, coreType="NumpyTensorCore"):
    if headType == "truthEvaluation":
        headValues = np.zeros(shape=(2))
        headValues[1] = 1  # weight
    elif headType == "weightedTruthEvaluation":
        headValues = np.zeros(shape=(2))
        headValues[1] = weight
    elif headType == "expFactor":
        headValues = create_expFactor_values(weight, False)
    elif headType == "diffExpFactor":
        headValues = create_expFactor_values(weight, True)
    else:
        raise ValueError("Headtype {} not understood!".format(headType))
    expressionString = get_expression_string(expression)
    return {expressionString + "_headCore": tensor.get_core(coreType=coreType)(headValues, [expressionString],
                                                                               expressionString + "_headCore")}


#####

## When only atoms in expressions (FormulaTensor)


def create_subExpressionCores(expression, formulaKey="", coreType="NumpyTensorCore"):
    addCoreKey = str(formulaKey) + "_" + str(expression) + "_subCore"
    headColor = str(formulaKey) + "_" + str(expression)

    if isinstance(expression, str):
        return {addCoreKey: tensor.get_core(coreType=coreType)(np.eye(2), [expression, headColor], addCoreKey)}
    elif len(expression) == 2:
        if expression[0] != "not":
            raise ValueError("Expression {} not understood!".format(expression))
        if isinstance(expression[1], str):
            return {addCoreKey: tensor.get_core(coreType=coreType)(create_negation_tensor(),
                                                                   [expression[1], headColor],
                                                                   addCoreKey)}
        else:
            partsDict = create_subExpressionCores(expression[1], formulaKey)
            addCore = tensor.get_core(coreType=coreType)(create_negation_tensor(),
                                                         [formulaKey + "_" + str(expression[1]), headColor], addCoreKey)
            return {**partsDict, addCoreKey: addCore}
    elif len(expression) == 3:
        if isinstance(expression[0], str):
            partsDict0 = {}
            leftColor = expression[0]
        else:
            partsDict0 = create_subExpressionCores(expression[0], formulaKey)
            leftColor = formulaKey + "_" + str(expression[0])

        if isinstance(expression[2], str):
            partsDict2 = {}
            rightColor = expression[2]
        else:
            partsDict2 = create_subExpressionCores(expression[2], formulaKey)
            rightColor = formulaKey + "_" + str(expression[2])

        return {**partsDict0, **partsDict2,
                addCoreKey: tensor.get_core(coreType=coreType)(get_binary_tensor(expression[1]),
                                                               [leftColor, rightColor, headColor],
                                                               addCoreKey)}

    else:
        raise ValueError("Expression {} has wrong length!".format(expression))


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


def create_expFactor_values(weight, differentiated):
    values = np.zeros(shape=(2))
    if not differentiated:
        values[0] = 1
    values[1] = np.exp(weight)
    return values


def create_emptyCoresDict(variableList):
    return {variableKey + "_trivialCore": tensor.get_core(coreType=coreType)(np.ones(2), [variableKey],
                                                                             variableKey + "_trivialCore")
            for variableKey in variableList}


def create_evidenceCoresDict(evidenceDict, coreType="NumpyTensorCore"):
    evidenceCoresDict = {}
    for atomKey in evidenceDict:
        truthValues = np.zeros(shape=(2))
        if bool(evidenceDict[atomKey]):
            truthValues[1] = 1
        else:
            truthValues[0] = 1
        evidenceCoresDict[atomKey + "_evidence"] = tensor.get_core(coreType=coreType)(truthValues, [atomKey],
                                                                                      atomKey + "_evidence")
    return evidenceCoresDict


## When Placeholders in Expression (SuperposedFormulaTensor)
def skeleton_recursion(headExpression, candidatesDict):
    if type(headExpression) == str:
        return {}, candidatesDict[headExpression]
    elif headExpression[0] == "not":
        if type(headExpression[1]) == str:
            return create_negationCoreDict(candidatesDict[headExpression[1]], inprefix=str(headExpression[1]) + "_",
                                           outprefix=str(headExpression) + "_"), candidatesDict[headExpression[1]]
        else:
            skeletonCoresDict, atoms = skeleton_recursion(headExpression[1], candidatesDict)
            return {**skeletonCoresDict,
                    **create_negationCoreDict(atoms, inprefix=str(headExpression[1]) + "_",
                                              outprefix=str(headExpression) + "_")}, atoms
    elif headExpression[1] == "and":
        if type(headExpression[0]) == str:
            leftskeletonCoresDict = {headExpression[0] + "_" + atomKey + "_l": create_deltaCore(
                colors=[headExpression[0] + "_" + atomKey, str(headExpression) + "_" + atomKey])
                for atomKey in candidatesDict[headExpression[0]]}
            leftatoms = candidatesDict[headExpression[0]]
        else:
            leftskeletonCoresDict, leftatoms = skeleton_recursion(headExpression[0], candidatesDict)

            leftskeletonCoresDict = {**leftskeletonCoresDict,
                                     **{str(headExpression[0]) + "_" + atomKey + "_lPass": create_deltaCore(
                                         [str(headExpression[0]) + "_" + atomKey, str(headExpression) + "_" + atomKey])
                                         for atomKey in leftatoms}
                                     }
        if type(headExpression[2]) == str:
            rightskeletonCoresDict = {headExpression[2] + "_" + atomKey + "_r": create_deltaCore(
                colors=[headExpression[2] + "_" + atomKey, str(headExpression) + "_" + atomKey])
                for atomKey in candidatesDict[headExpression[2]]}
            rightatoms = candidatesDict[headExpression[2]]
        else:
            rightskeletonCoresDict, rightatoms = skeleton_recursion(headExpression[2], candidatesDict)
            rightskeletonCoresDict = {**rightskeletonCoresDict,
                                      **{str(headExpression[2]) + "_" + atomKey + "_rPass": create_deltaCore(
                                          [str(headExpression[2]) + "_" + atomKey, str(headExpression) + "_" + atomKey])
                                          for atomKey in rightatoms}
                                      }
        return {**leftskeletonCoresDict, **rightskeletonCoresDict}, leftatoms + rightatoms


def create_negationCoreDict(atoms, inprefix, outprefix):
    negationMatrix = np.zeros(shape=(2, 2))
    negationMatrix[0, 1] = 1
    negationMatrix[1, 0] = 1

    negationCoreDict = {}
    for atomKey in atoms:
        negationCoreDict[outprefix + "_" + atomKey + "_neg"] = tensor.get_core(coreType=coreType)(negationMatrix,
                                                                                                  [inprefix + atomKey,
                                                                                                   outprefix + atomKey],
                                                                                                  outprefix + atomKey + "_neg")
    return negationCoreDict


def create_deltaCore(colors, name=""):
    values = np.zeros(shape=[2 for i in range(len(colors))])
    values[tuple(0 for color in colors)] = 1
    values[tuple(1 for color in colors)] = 1
    return tensor.get_core(coreType=coreType)(values, colors, name)


def create_selectorCoresDict(candidatesDict):
    ## incolors: placeHolderKey
    ## outcolors: placeHolderKey + "_" + atomKey
    selectorCoresDict = {}
    for placeHolderKey in candidatesDict:
        selectorCoresDict = {**selectorCoresDict,
                             **create_local_selectorCores(candidatesDict[placeHolderKey], placeHolderKey)}
    return selectorCoresDict


def create_local_selectorCores(atoms, placeHolderKey):
    returnDict = {}
    for i, atomKey in enumerate(atoms):
        coreValues = np.ones(shape=(len(atoms), 2))
        coreValues[i, 0] = 0
        returnDict[placeHolderKey + "_" + atomKey + "_selector"] = tensor.get_core(coreType=coreType)(
            coreValues, [placeHolderKey, placeHolderKey + "_" + atomKey],
            placeHolderKey + "_" + atomKey + "_selector")
    return returnDict


## DataCore Creation
def dataCore_from_sampleDf(sampleDf, atomKey, dataColor):
    if atomKey not in sampleDf.keys():
        raise ValueError
    dfEntries = sampleDf[atomKey].values
    dataNum = dfEntries.shape[0]
    values = np.zeros(shape=(dataNum, 2))
    for i in range(dataNum):
        if dfEntries[i] == 0:
            values[i, 0] = 1
        else:
            values[i, 1] = 1
    return tensor.get_core(coreType=coreType)(values, [dataColor, atomKey])


## ConstraintCore Creation:
def create_constraintCoresDict(atoms, name):
    constraintCoresDict = {}
    for i, atomKey in enumerate(atoms):
        coreValues = np.zeros(shape=(len(atoms), 2))
        coreValues[:, 0] = np.ones(shape=(len(atoms)))
        coreValues[i, 0] = 0
        coreValues[i, 1] = 1
        constraintCoresDict[name + "_" + atomKey + "_cconstraint"] = tensor.get_core(coreType=coreType)(
            core_values=coreValues,
            core_colors=[
                name + "_cconstraint",
                atomKey],
            name=name + "_" + atomKey + "_cconstraint")
    return constraintCoresDict


## Analysis

## Check whether the colors in all coreDicts match wrt each other and the knownShapesDict
def check_colorShapes(coresDicts, knownShapesDict={}):
    for coresDict in coresDicts:
        for coreKey in coresDict:
            for i, color in enumerate(coresDict[coreKey].colors):
                coreColorShape = coresDict[coreKey].values.shape[i]
                if color not in knownShapesDict:
                    knownShapesDict[color] = coreColorShape
                else:
                    if knownShapesDict[color] != coreColorShape:
                        raise ValueError("Core {} has unexpected shape of color {}.".format(coreKey, color))
