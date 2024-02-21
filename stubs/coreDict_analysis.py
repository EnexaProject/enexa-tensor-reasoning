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
