defaultCoreType = "NumpyTensorCore"


def get_core(coreType=defaultCoreType):
    if coreType == "NumpyTensorCore":
        from tnreason.engine.workload_to_numpy import NumpyCore
        return NumpyCore
    elif coreType == "PolynomialCore":
        from tnreason.engine.polynomial_contractor import PolynomialCore
        return PolynomialCore
    else:
        raise ValueError("Core Type {} not supported.".format(coreType))


def create_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType=defaultCoreType,
                               name="Encoding"):
    if coreType == "NumpyTensorCore":
        from tnreason.engine.workload_to_numpy import np_rencoding_from_function
        return np_rencoding_from_function(inshape, outshape, incolors, outcolors, function, name)
    elif coreType == "PolynomialCore":
        from tnreason.engine.polynomial_contractor import poly_rencoding_from_function
        return poly_rencoding_from_function(inshape, outshape, incolors, outcolors, function, name)
    else:
        raise ValueError("Core Type {} not supported for .".format(coreType))


def reduce_function(function, coordinates):
    return lambda x: [function(x)[coordinate] for coordinate in coordinates]


def create_partitioned_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType=defaultCoreType,
                                           partitionDict=None, nameSuffix="_encodingCore"):
    if partitionDict is None:
        partitionDict = {color: [color] for color in outcolors}
    indDict = {}
    for parKey in partitionDict:
        indDict[parKey] = [outcolors.index(outcolor) for outcolor in partitionDict[parKey]]
    return {parKey + nameSuffix:
                create_relational_encoding(inshape=inshape,
                                           outshape=[outshape[i] for i in indDict[parKey]],
                                           incolors=incolors,
                                           outcolors=partitionDict[parKey],
                                           function=reduce_function(function, indDict[parKey]),
                                           coreType=coreType,
                                           name=parKey + nameSuffix)
            for parKey in partitionDict}
