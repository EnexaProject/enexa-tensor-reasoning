from tnreason.logic import coordinate_calculus as cc
from tnreason.logic import basis_calculus as bc

import numpy as np

def create_formulaProcedure(expression, formulaKey):
    addCoreKey = formulaKey + "_" + str(expression) + "_"
    if type(expression) == str:
        return {addCoreKey: cc.CoordinateCore(np.eye(2),[expression,expression+"_h"],expression)}
    elif expression[0] == "not":
        if type(expression[1])==str:
            return {addCoreKey: cc.CoordinateCore(bc.create_negation_tensor(),[expression[1], str(expression) +"_h"],expression)}
        else:
            partsDict = create_formulaProcedure(expression[1],formulaKey)
            partsDict[addCoreKey] = cc.CoordinateCore(bc.create_negation_tensor(),[expression[1], str(expression) +"_h"], expression)
            return partsDict
    elif expression[1] == "and":
        if type(expression[0])==str:
            partsDict0 = {}
            leftColor = expression[0]
        else:
            partsDict0 = create_formulaProcedure(expression[0],formulaKey)
            leftColor = str(expression[0])+"_h"

        if type(expression[2])==str:
            prePartsDict2 = {}
            rightColor = expression[2]
        else:
            prePartsDict2 = create_formulaProcedure(expression[2],formulaKey)
            rightColor = str(expression[2])+"_h"

        ## STILL PROBLEM: SAME KEYS MIGHT APPEAR ON LEFT AND RIGHT SIDE -> Need to shift colors by adding zeros until not appearing !
        partsDict2 = {}
        for key in prePartsDict2:
            if key in partsDict0:
                partsDict2[key+"0"] = prePartsDict2[key]

        colors0 = get_colors_from_coreDict(partsDict0)
        preColors2 = get_colors_from_coreDict(partsDict2)
        replaceColorDict = create_newColorDict(colors0, preColors2)
        partsDict2 = replace_colors_in_coreDict(partsDict2, replaceColorDict)

        if rightColor in replaceColorDict:
            rightColor = replaceColorDict[rightColor]

        print(replaceColorDict)

        partsDict = {**partsDict0, **partsDict2}
        partsDict[addCoreKey] = cc.CoordinateCore(bc.create_and_tensor(),[leftColor, rightColor, str(expression) + "_h"], str(expression))
        return partsDict


def get_colors_from_coreDict(coreDict):
    colors = []
    for coreKey in coreDict:
        for color in coreDict[coreKey].colors:
            if color not in colors:
                colors.append(color)
    return colors

def create_newColorDict(colorsLeft, colorsRight):
    newColorDict = {}
    for color in colorsRight:
        if color in colorsLeft and "_h" in color:
            newColor = color
            while newColor in colorsLeft:
                newColor = newColor + "0"
            newColorDict[color] = newColor
    return newColorDict

def replace_colors_in_coreDict(coreDict, newColorDict):
    for coreKey in coreDict:
        for i, color in enumerate(coreDict[coreKey].colors):
            if color in newColorDict:
                coreDict[coreKey].colors[i] = newColorDict[color]
    return coreDict

if __name__ == "__main__":
    expression = [["not","sledz"],"and",["not","sledz"]]

    solDict  = create_formulaProcedure(expression,"Sledz")
    print([solDict[coreKey].colors for coreKey in solDict])
