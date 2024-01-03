from tnreason.logic import expression_utils as eu
from tnreason.logic import coordinate_calculus as cc

from tnreason.contraction import layers as lay

import numpy as np

def generate_negation_layer(skeletonExpression, shapesDict, sampleNum):
    variableColors = eu.get_variables(skeletonExpression)
    variableShapes = [shapesDict[color] for color in variableColors]

    directCore = cc.CoordinateCore(np.eye(sampleNum), [str(skeletonExpression[0]), str(skeletonExpression)])
    constantCore = cc.CoordinateCore(np.ones(shape=variableShapes+[sampleNum]), variableColors + [str(skeletonExpression)])

    return lay.Layer({str(skeletonExpression)+"_"+"constant": constantCore,
                      str(skeletonExpression)+"_"+"direct": directCore},
                     {str(skeletonExpression) + "_" + "constant": 1,
                      str(skeletonExpression) + "_" + "direct": -1})

def generate_conjunction_layer(skeletonExpression, sampleNum):
    values = np.empty(shape=(sampleNum, sampleNum, sampleNum))
    for i in range(sampleNum):
        values[i, i, i] = 1
    conjunctionCore = cc.CoordinateCore(values,[str(skeletonExpression[0]),str(skeletonExpression[2]),str(skeletonExpression)])
    return lay.Layer({str(skeletonExpression): conjunctionCore},
                     {str(skeletonExpression): 1})

def generate_skeletonLayerDict(skeletonExpression, shapesDict, sampleNum):
    addCoreKey = str(skeletonExpression)
    if type(skeletonExpression) == str:
        return {}
    elif skeletonExpression[0] == "not":
        layerDict = generate_skeletonLayerDict(skeletonExpression[1], shapesDict, sampleNum)
        layerDict[addCoreKey] = generate_negation_layer(skeletonExpression, shapesDict, sampleNum)
        return layerDict
    elif skeletonExpression[1] == "and":
        layerDict = {**generate_skeletonLayerDict(skeletonExpression[0], shapesDict, sampleNum),
                     **generate_skeletonLayerDict(skeletonExpression[2], shapesDict, sampleNum)}
        layerDict[addCoreKey] = generate_conjunction_layer(skeletonExpression, sampleNum)
        return layerDict

if __name__ == "__main__":
    print(generate_skeletonLayerDict(["a","and",["not","b"]], {"a": 2, "b": 10}, 4).keys())
    print(generate_skeletonLayerDict(["a","and",["not","b"]], {"a": 2, "b": 10}, 4)["['not', 'b']"].colors)