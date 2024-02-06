from tnreason.logic import coordinate_calculus as cc

import numpy as np


## When only atoms in expressions (FormulaTensor)
def create_subExpressionCores(expression, formulaKey):
    addCoreKey = str(formulaKey) + "_" + str(expression) + "_subCore"
    headColor = str(formulaKey) + "_" + str(expression)

    if isinstance(expression, str):
        return {addCoreKey: cc.CoordinateCore(np.eye(2), [expression, headColor], addCoreKey)}
    elif len(expression) == 2:
        if expression[0] != "not":
            raise ValueError("Expression {} not understood!".format(expression))
        if isinstance(expression[1], str):
            return {addCoreKey: cc.CoordinateCore(create_negation_tensor(),
                                                  [expression[1], headColor],
                                                  addCoreKey)}
        else:
            partsDict = create_subExpressionCores(expression[1], formulaKey)
            addCore = cc.CoordinateCore(create_negation_tensor(),
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

        if expression[1] == "and":
            addCoreValue = create_conjunction_tensor()
        elif expression[1] == "or":
            addCoreValue = create_disjunction_tensor()
        elif expression[1] == "xor":
            addCoreValue = create_xor_tensor()
        elif expression[1] == "imp":
            addCoreValue = create_implication_tensor()
        elif expression[1] == "eq":
            addCoreValue = create_biconditional_tensor()
        else:
            raise ValueError("Expression {} not understood!")
        return {**partsDict0, **partsDict2,
                addCoreKey: cc.CoordinateCore(addCoreValue, [leftColor, rightColor, headColor], addCoreKey)}

    else:
        raise ValueError("Expression {} has wrong length!".format(expression))


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


def create_headCore(headType, weight, headColor):
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
    return cc.CoordinateCore(headValues, [headColor])


def create_expFactor_values(weight, differentiated):
    values = np.zeros(shape=(2))
    if not differentiated:
        values[0] = 1
    values[1] = np.exp(weight)
    return values

def create_emptyCoresDict(variableList):
    return {variableKey+"_trivialCore": cc.CoordinateCore(np.ones(2), [variableKey], variableKey + "_trivialCore")
            for variableKey in variableList}

def create_evidenceCoresDict(evidenceDict):
    evidenceCoresDict = {}
    for atomKey in evidenceDict:
        truthValues = np.zeros(shape=(2))
        if bool(evidenceDict[atomKey]):
            truthValues[1] = 1
        else:
            truthValues[0] = 1
        evidenceCoresDict[atomKey + "_evidence"] = cc.CoordinateCore(truthValues, [atomKey], atomKey + "_evidence")
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
        negationCoreDict[outprefix + "_" + atomKey + "_neg"] = cc.CoordinateCore(negationMatrix,
                                                                                 [inprefix + atomKey,
                                                                                  outprefix + atomKey],
                                                                                 outprefix + atomKey + "_neg")
    return negationCoreDict


def create_deltaCore(colors, name=""):
    values = np.zeros(shape=[2 for i in range(len(colors))])
    values[tuple(0 for color in colors)] = 1
    values[tuple(1 for color in colors)] = 1
    return cc.CoordinateCore(values, colors, name)


def create_selectorCoresDict(candidatesDict):
    ## incolors: placeHolderKey
    ## outcolors: placeHolderKey + "_" + atomKey
    selectorCoresDict = {}
    for placeHolderKey in candidatesDict:
        for i, atomKey in enumerate(candidatesDict[placeHolderKey]):
            coreValues = np.ones(shape=(len(candidatesDict[placeHolderKey]), 2))
            coreValues[i, 0] = 0
            selectorCoresDict[placeHolderKey + "_" + atomKey + "_selector"] = cc.CoordinateCore(
                coreValues, [placeHolderKey, placeHolderKey + "_" + atomKey],
                placeHolderKey + "_" + atomKey + "_selector")
    return selectorCoresDict


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
    return cc.CoordinateCore(values, [dataColor, atomKey])


## ConstraintCore Creation:
def create_constraintCoresDict(atoms, name):
    constraintCoresDict = {}
    for i, atomKey in enumerate(atoms):
        coreValues = np.zeros(shape=(len(atoms),2))
        coreValues[:,0] = np.ones(shape=(len(atoms)))
        coreValues[i,0] = 0
        coreValues[i,1] = 1
        constraintCoresDict[name+"_"+atomKey+"_cconstraint"] = cc.CoordinateCore(core_values=coreValues,
                                                                                 core_colors=[name+"_cconstraint", atomKey],
                                                                                 name=name+"_"+atomKey+"_cconstraint")
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
