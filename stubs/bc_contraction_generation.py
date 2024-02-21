from tnreason.logic import coordinate_calculus as cc
from tnreason.logic import basis_calculus as bc

import numpy as np


def generate_rawCoreDict(expressionDict):
    ## ExpressionDict of structure key: expression
    rawCoreDict = {}
    for formulaKey in expressionDict:
        expression = expressionDict[formulaKey]
        rawCoreDict = {**rawCoreDict, **generate_factor_dict(expression, formulaKey=formulaKey, headType="empty")}
    return rawCoreDict


def generate_formulaCoreDict(expressionDict):
    ## ExpressionDict of structure key: [expression, weight]
    rawCoreDict = {}
    for formulaKey in expressionDict:
        expression = expressionDict[formulaKey][0]
        weight = expressionDict[formulaKey][1]
        rawCoreDict = {**rawCoreDict,
                       **generate_factor_dict(expression, formulaKey=formulaKey, weight=weight, headType="expFactor")}
    return rawCoreDict


def generate_exponentiationHeadValues(weight, headcolors, differentiated=False):
    if differentiated:
        values = np.zeros(shape=2)
    else:
        values = np.ones(shape=2)
    values[1] = np.exp(weight)
    return cc.CoordinateCore(values, headcolors)


def generate_factor_dict(expression, formulaKey="f0", weight=0, headType="truthEvaluation"):
    factorDict = create_formulaProcedure(expression, formulaKey)
    headColors = [formulaKey + "_" + str(expression)]
    if headType == "truthEvaluation":
        factorDict[formulaKey + "_" + str(expression) + "_" + headType] = cc.CoordinateCore(
            bc.create_truth_vec(), headColors)
    elif headType == "expFactor":
        factorDict[formulaKey + "_" + str(expression) + "_" + headType] = generate_exponentiationHeadValues(weight,
                                                                                                            headColors,
                                                                                                            differentiated=False)
    elif headType == "diffExpFactor":
        factorDict[formulaKey + "_" + str(expression) + "_" + headType] = generate_exponentiationHeadValues(weight,
                                                                                                            headColors,
                                                                                                            differentiated=True)
    elif headType == "empty":
        pass
    else:
        raise ValueError("Head Type {} not understood!".format(headType))
    return factorDict


## For the generation of Basis Calculus Instructions
def create_formulaProcedure(expression, formulaKey):
    addCoreKey = str(formulaKey) + "_" + str(expression) + "_subCore"
    if type(expression) == str:
        return {addCoreKey: cc.CoordinateCore(np.eye(2), [expression, formulaKey + "_" + expression], expression)}
    elif expression[0] == "not":
        if type(expression[1]) == str:
            return {addCoreKey: cc.CoordinateCore(bc.create_negation_tensor(),
                                                  [expression[1], formulaKey + "_" + str(expression)],
                                                  expression)}
        else:
            partsDict = create_formulaProcedure(expression[1], formulaKey)
            partsDict[addCoreKey] = cc.CoordinateCore(bc.create_negation_tensor(),
                                                      [formulaKey + "_" + str(expression[1]),
                                                       formulaKey + "_" + str(expression)], expression)
            return partsDict
    elif expression[1] == "and":
        if type(expression[0]) == str:
            partsDict0 = {}
            leftColor = expression[0]
        else:
            partsDict0 = create_formulaProcedure(expression[0], formulaKey)
            leftColor = formulaKey + "_" + str(expression[0])

        if type(expression[2]) == str:
            partsDict2 = {}
            rightColor = expression[2]
        else:
            partsDict2 = create_formulaProcedure(expression[2], formulaKey)
            rightColor = formulaKey + "_" + str(expression[2])

        return {**partsDict0, **partsDict2,
                addCoreKey: cc.CoordinateCore(bc.create_and_tensor(),
                                              [leftColor, rightColor, formulaKey + "_" + str(expression)],
                                              str(expression))}
        ## OLD: Resolving key conflicts in the dictionary
        # NOT NEEDED! If colliding, then same binary core

        ## Renaming cores of the right hand side to avoid key collision
        # partsDict2 = {}
        # for key in prePartsDict2:
        #    if key in partsDict0:
        #        partsDict2[key + "0"] = prePartsDict2[key]
        #    else:
        #        partsDict2[key + "0"] = prePartsDict2[key]

        ## Renaming colors of the right hand side to avoid duplicates (except for atoms)
        # colors0 = get_colors_from_coreDict(partsDict0)
        # preColors2 = get_colors_from_coreDict(partsDict2)
        # replaceColorDict = create_newColorDict(colors0, preColors2)
        # partsDict2 = replace_colors_in_coreDict(partsDict2, replaceColorDict)
        # if rightColor in replaceColorDict:
        #    rightColor = replaceColorDict[rightColor]


# def get_colors_from_coreDict(coreDict):
#     colors = []
#     for coreKey in coreDict:
#         for color in coreDict[coreKey].colors:
#             if color not in colors:
#                 colors.append(color)
#     return colors
#
#
# def create_newColorDict(colorsLeft, colorsRight):
#     newColorDict = {}
#     for color in colorsRight:
#         if color in colorsLeft and "_h" in color:
#             newColor = color
#             while newColor in colorsLeft:
#                 newColor = newColor + "0"
#             newColorDict[color] = newColor
#     return newColorDict
#
#
# def replace_colors_in_coreDict(coreDict, newColorDict):
#     for coreKey in coreDict:
#         for i, color in enumerate(coreDict[coreKey].colors):
#             if color in newColorDict:
#                 coreDict[coreKey].colors[i] = newColorDict[color]
#     return coreDict


if __name__ == "__main__":
    expression = [["not", "sledz"], "and", ["not", "sledz"]]

    solDict = create_formulaProcedure(expression, "Sledz")
    print([solDict[coreKey].colors for coreKey in solDict])
