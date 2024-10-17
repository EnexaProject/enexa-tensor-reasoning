defaultCoreType = "NumpyTensorCore"


def get_core(coreType=None):
    if coreType is None:
        coreType = defaultCoreType
    if coreType == "NumpyTensorCore":
        from tnreason.engine.workload_to_numpy import NumpyCore
        return NumpyCore
    elif coreType == "PolynomialCore":
        from tnreason.engine.polynomial_contractor import PolynomialCore
        return PolynomialCore
    elif coreType == "HypertrieCore":
        from tnreason.engine.workload_to_tentris import HypertrieCore
        return HypertrieCore
    else:
        raise ValueError("Core Type {} not supported.".format(coreType))


def create_tensor_encoding(inshape, incolors, function, coreType=None, name="Encoding"):
    if coreType is None:
        coreType = defaultCoreType
    if coreType == "NumpyTensorCore":
        from tnreason.engine.workload_to_numpy import np_tencoding_from_function
        return np_tencoding_from_function(inshape, incolors, function, name)
    elif coreType == "PolynomialCore":
        from tnreason.engine.polynomial_contractor import poly_tencoding_from_function
        return poly_tencoding_from_function(inshape, incolors, function, name)
    elif coreType == "HypertrieCore":
        from tnreason.engine.workload_to_tentris import ht_tencoding_from_function
        return ht_tencoding_from_function(inshape, incolors, function, name)
    else:
        raise ValueError("Core Type {} not supported for .".format(coreType))


def create_random_core(name, shape, colors,
                       randomEngine="NumpyUniform"):  # Works only for numpy cores! (do not have a random engine else)
    from tnreason.engine.workload_to_numpy import np_random_core
    return np_random_core(shape, colors, randomEngine, name)


def create_relational_encoding(inshape, outshape, incolors, outcolors, function, coreType=None,
                               name="Encoding"):
    """
    Creates relational encoding of a function as a single core.
    The function has to be a map from the indices in inshape to the indices in outshape.
    """
    if coreType is None:
        coreType = defaultCoreType
    if coreType == "NumpyTensorCore":
        from tnreason.engine.workload_to_numpy import np_rencoding_from_function
        return np_rencoding_from_function(inshape, outshape, incolors, outcolors, function, name)
    elif coreType == "PolynomialCore":
        from tnreason.engine.polynomial_contractor import poly_rencoding_from_function
        return poly_rencoding_from_function(inshape, outshape, incolors, outcolors, function, name)
    elif coreType == "HypertrieCore":
        from tnreason.engine.workload_to_tentris import ht_rencoding_from_function
        return ht_rencoding_from_function(inshape, outshape, incolors, outcolors, function, name)
    else:
        raise ValueError("Core Type {} not supported for .".format(coreType))


def get_image(core, inShape, imageValues=[float(0), float(1)]):
    import numpy as np
    for indices in np.ndindex(tuple(inShape)):
        coordinate = float(core[indices])
        if coordinate not in imageValues:
            imageValues.append(coordinate)
    return imageValues


def core_to_relational_encoding(core, headColor, outCoreType=None):
    imageValues = get_image(core, core.values.shape)
    return create_relational_encoding(inshape=core.values.shape, outshape=[len(imageValues)], incolors=core.colors,
                                      outcolors=[headColor], function=lambda *args: [imageValues.index(core[args])],
                                      coreType=outCoreType), imageValues


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
