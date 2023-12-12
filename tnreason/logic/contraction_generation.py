from tnreason.logic import coordinate_calculus as cc
from tnreason.logic import basis_calculus as bc

import numpy as np

def create_formulaProcedure(expression, formulaKey):
    addCoreKey = formulaKey + "_" + expression + "_"
    if type(expression) == str:
        return {addCoreKey: cc.CoordinateCore(np.eye(2),[expression,expression+"_h"],expression)}
    elif expression[0] == "not":
        if type(expression[1])==str:
            return {addCoreKey: cc.CoordinateCore(bc.create_negation_tensor(),[expression[1], expression+"_h"],expression)}
        else:
            partsDict = create_formulaProcedure(expression[1],formulaKey)
            partsDict[addCoreKey] = cc.CoordinateCore(bc.create_negation_tensor(),[expression[1], expression+"_h"], expression)
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
        #partsDict2 = {}
        #for key in prePartsDict2:
        #    if key in partsDict0:
        #        partsDict2[key+"0"] = cc.CoordinateCore(prePartsDict2[key],[color +"0" for color in prePartsDict2[key]])

        partsDict = partsDict0 + prePartsDict2
        partsDict[addCoreKey] = cc.CoordinateCore(bc.create_and_tensor(),[leftColor, rightColor, str(expression) + "_h"], str(expression))
        return partsDict

if __name__ == "__main__":
    solDict  = create_formulaProcedure("Sledz","Sledz")
    print([solDict[coreKey].colors for coreKey in solDict])
    print(create_formulaProcedure("Sledz","Sledz"))