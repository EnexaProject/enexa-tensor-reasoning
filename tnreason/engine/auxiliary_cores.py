from tnreason import engine

trivialCoreSuffix = "_trivialCore"


def create_trivial_cores(rawKeys, shapeDict=None, suffix=trivialCoreSuffix, coreType=None):
    """
    Creates dictionary of trivial cores with coordinate 1, which act as neutral placeholders in contractions.
        * rawKeys: List of raw keys (added by suffix)
        * shapeDict: Dictionary of shapes of the trivial core to each rawKey, default: 2
    """
    if shapeDict is None:
        shapeDict = {key: 2 for key in rawKeys}
    return {key + suffix: create_trivial_core(key + suffix, shapeDict[key], [key], coreType=coreType) for key in
            rawKeys}


def create_trivial_core(name, shape, colors, coreType=None):
    return engine.create_tensor_encoding(inshape=shape, incolors=colors, function=lambda *args: 1, coreType=coreType,
                                         name=name)


def create_basis_core(name, shape, colors, numberTuple, coreType=None):
    if isinstance(numberTuple, tuple) or isinstance(numberTuple, list):
        numberTuple = tuple([int(number) for number in numberTuple])
    else:  # Dealing with np.int, Booleans, Floats
        numberTuple = tuple([int(numberTuple)])
    return engine.create_tensor_encoding(inshape=shape, incolors=colors,
                                         function=lambda *args: int(args == numberTuple), coreType=coreType, name=name)
