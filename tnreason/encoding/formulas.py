from tnreason import engine

import numpy as np

from tnreason.encoding import connectives as encon


def create_formulas_cores(specDict, alreadyCreated=[]):
    knowledgeCores = {}
    for formulaName in specDict.keys():
        if isinstance(specDict[formulaName][-1], float) or isinstance(specDict[formulaName][-1], int):
            knowledgeCores = {**knowledgeCores,
                              **create_headCore(get_formula_color(specDict[formulaName][0]), "expFactor", weight=
                              specDict[formulaName][1]),
                              **create_raw_formula_cores(specDict[formulaName][0],
                                                         alreadyCreated=
                                               list(knowledgeCores.keys()) + alreadyCreated)}
        else:
            knowledgeCores = {**knowledgeCores,
                              **create_headCore(get_formula_color(specDict[formulaName]), "truthEvaluation"),
                              **create_raw_formula_cores(specDict[formulaName],
                                                         alreadyCreated=list(knowledgeCores.keys()) + alreadyCreated)}
    return knowledgeCores


def create_conCore(expression, coreType="NumpyTensorCore"):
    expressionString = get_formula_color(expression)
    if isinstance(expression, str):
        return {}
    elif len(expression) == 2:
        preExpressionString = get_formula_color(expression[1])
        return {expressionString + "_conCore": engine.get_core(coreType=coreType)(
            encon.get_unary_tensor(expression[0]),
            [preExpressionString, expressionString],
            expressionString + "_conCore")
        }

    elif len(expression) == 3:
        leftExpressionString = get_formula_color(expression[0])
        rightExpressionString = get_formula_color(expression[2])
        return {
            expressionString + "_conCore": engine.get_core(coreType=coreType)(encon.get_binary_tensor(expression[1]),
                                                                              [leftExpressionString,
                                                                               rightExpressionString,
                                                                               expressionString],
                                                                              expressionString + "_conCore")}
    else:
        raise ValueError("Expression {} not understood!".format(expression))


def create_raw_formula_cores(expression, coreType="NumpyTensorCore", alreadyCreated=[]):
    if get_formula_color(expression) + "_conCore" in alreadyCreated:
        return {}
    if isinstance(expression, str):
        return create_conCore(expression, coreType=coreType)
    elif len(expression) == 2:
        return {**create_conCore(expression, coreType=coreType),
                **create_raw_formula_cores(expression[1], coreType=coreType, alreadyCreated=alreadyCreated)}
    elif len(expression) == 3:
        return {**create_conCore(expression, coreType=coreType),
                **create_raw_formula_cores(expression[0], coreType=coreType, alreadyCreated=alreadyCreated),
                **create_raw_formula_cores(expression[2], coreType=coreType, alreadyCreated=alreadyCreated)
                }


def create_headCore(expression, headType, weight=None, coreType="NumpyTensorCore", name=None):
    if headType == "truthEvaluation":
        headValues = np.zeros(shape=(2))
        headValues[1] = 1
    elif headType == "falseEvaluation":
        headValues = np.zeros(shape=(2))
        headValues[0] = 1
    elif headType == "weightedTruthEvaluation":
        headValues = np.zeros(shape=(2))
        headValues[1] = weight
    elif headType == "expFactor":
        headValues = create_expFactor_values(weight, False)
    elif headType == "diffExpFactor":
        headValues = create_expFactor_values(weight, True)
    else:
        raise ValueError("Headtype {} not understood!".format(headType))

    color = get_formula_color(expression)

    if name is None:
        name = color + "_headCore"

    return {name: engine.get_core(coreType=coreType)(headValues, [color],
                                                     name)}


def create_expFactor_values(weight, differentiated):
    values = np.zeros(shape=(2))
    if not differentiated:
        values[0] = 1
    values[1] = np.exp(weight)
    return values


def get_formula_color(expression):
    if isinstance(expression, str):
        return expression
    elif len(expression) == 2:
        assert isinstance(expression[0], str)
        return expression[0] + "_" + get_formula_color(expression[1])
    elif len(expression) == 3:
        assert isinstance(expression[1], str)
        return "(" + get_formula_color(expression[0]) + "_" + expression[1] + "_" + get_formula_color(
            expression[2]) + ")"
