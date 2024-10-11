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


def create_tensor_encoding(inshape, incolors, function, coreType=defaultCoreType, name="Encoding"):
    if coreType == "NumpyTensorCore":
        from tnreason.engine.workload_to_numpy import np_tencoding_from_function
        return np_tencoding_from_function(inshape, incolors, function, name)
    elif coreType == "PolynomialCore":
        from tnreason.engine.polynomial_contractor import poly_tencoding_from_function
        return poly_tencoding_from_function(inshape, incolors, function, name)
    else:
        raise ValueError("Core Type {} not supported for .".format(coreType))


def create_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType=defaultCoreType,
                               name="Encoding"):
    """
    Creates relational encoding of a function as a single core.
    """
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
    """
    Creates relational encoding of a function as a tensor network, where the output axis are splitted according to the partionDict.
    """
    if partitionDict is None:
        partitionDict = {color: [color] for color in outcolors}
    return {parKey + nameSuffix:
                create_relational_encoding(inshape=inshape,
                                           outshape=[outshape[outcolors.index(c)] for c in partitionDict[parKey]],
                                           incolors=incolors,
                                           outcolors=partitionDict[parKey],
                                           function=lambda x: [function(x)[outcolors.index(c)] for c in
                                                               partitionDict[parKey]],
                                           coreType=coreType,
                                           name=parKey + nameSuffix)
            for parKey in partitionDict}
