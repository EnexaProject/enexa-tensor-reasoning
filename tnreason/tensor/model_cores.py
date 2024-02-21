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
        return "(" + get_expression_string(expression[0]) + "_" + expression[1] + "_" + get_expression_string(
            expression[2]) + ")"


def create_conCore(expression, coreType="NumpyTensorCore"):
    expressionString = get_expression_string(expression)

    if isinstance(expression, str):
        return {}
        #expressionString + "_conCore": tensor.get_core(coreType=coreType)(np.ones(2), [expressionString],
        #                                                                          expressionString + "_conCore")}
    elif len(expression) == 2:
        preExpressionString = get_expression_string(expression[1])
        return {expressionString + "_conCore": tensor.get_core(coreType=coreType)(
            get_unary_tensor(expression[0]),
            [preExpressionString, expressionString],
            expressionString + "_conCore")
        }

    elif len(expression) == 3:
        leftExpressionString = get_expression_string(expression[0])
        rightExpressionString = get_expression_string(expression[2])
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


##### Constraint Core Arrays

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


def create_emptyCoresDict(variableList, coreType="NumpyTensorCore", varDimDict=None):
    if varDimDict is None:
        varDimDict = {variableKey : 2 for variableKey in variableList}
    return {variableKey + "_trivialCore": tensor.get_core(coreType=coreType)(np.ones(varDimDict[variableKey]), [variableKey],
                                                                             variableKey + "_trivialCore")
            for variableKey in variableList}


def create_evidenceCoresDict(evidenceDict, dimDict=None,coreType="NumpyTensorCore"):
    if dimDict is None:
        dimDict = {evidenceKey : 2 for evidenceKey in evidenceDict}
    evidenceCoresDict = {}
    for atomKey in evidenceDict:
        truthValues = np.zeros(shape=(dimDict[atomKey]))
        if bool(evidenceDict[atomKey]):
            truthValues[1] = 1
        else:
            truthValues[0] = 1
        evidenceCoresDict[atomKey + "_evidence"] = tensor.get_core(coreType=coreType)(truthValues, [atomKey],
                                                                                      atomKey + "_evidence")
    return evidenceCoresDict

## DataCore Creation
def dataCore_from_sampleDf(sampleDf, atomKey, dataColor, coreType="NumpyTensorCore"):
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
def create_constraintCoresDict(atoms, name, coreType="NumpyTensorCore"):
    constraintCoresDict = {}
    for i, atomKey in enumerate(atoms):
        coreValues = np.zeros(shape=(len(atoms), 2))
        coreValues[:, 0] = np.ones(shape=(len(atoms)))
        coreValues[i, 0] = 0
        coreValues[i, 1] = 1
        constraintCoresDict[name + "_" + atomKey + "_catCore"] = tensor.get_core(coreType=coreType)(
            coreValues,
            [
                name,
                atomKey],
            name=name + "_" + atomKey + "_catCore")
    return constraintCoresDict