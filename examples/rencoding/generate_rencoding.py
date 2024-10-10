import numpy as np
from tnreason import engine


def generate_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType="NumpyTensorCore"):
    if coreType == "NumpyTensorCore":
        values = np.zeros(inshape + outshape)
        for i in np.ndindex(*inshape):
            values[i + tuple(
                [int(entry) for entry in function(*i)])] = 1  # Watch out: Numpy interpretes True as full slice!

    elif coreType == "PolynomialCore":
        values = []
        colors = incolors + outcolors
        for i in np.ndindex(*inshape):
            values.append((1, {colors[k]: assignment for k, assignment in
                               enumerate(i + tuple([int(entry) for entry in function(*i)]))}))
    else:
        raise ValueError("Core type not understood.")
    return engine.get_core(coreType=coreType)(values=values, colors=incolors + outcolors)


def reduce_function(function, coordinates):
    return lambda x: [function(x)[coordinate] for coordinate in coordinates]


def create_coreDict_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType="NumpyTensorCore",partitionDict=None):
    if partitionDict is None:
        partitionDict = {color : [color] for color in outcolors}
    indDict = {}
    for parKey in partitionDict:
        indDict[parKey] = [outcolors.index(outcolor) for outcolor in partitionDict[parKey]]

    return {parKey + "_encodingCore": generate_relational_encoding(inshape=inshape, outshape=[outshape[i] for i in indDict[parKey]],
                                                                     incolors=incolors, outcolors=partitionDict[parKey],
                                                                     function=reduce_function(function, indDict[parKey]),
                                                                     coreType=coreType)
            for parKey in partitionDict}
