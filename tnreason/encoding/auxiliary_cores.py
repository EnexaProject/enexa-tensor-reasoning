from tnreason import engine

import numpy as np

trivialCoreSuffix = "_trivialCore"


def create_trivial_cores(rawKeys, shapeDict=None, suffix=trivialCoreSuffix):
    """
    Creates dictionary of trivial cores with coordinate 1, which act as neutral placeholders in contractions.
        * rawKeys: List of raw keys (added by suffix)
        * shapeDict: Dictionary of shapes of the trivial core to each rawKey, default: 2
    """
    if shapeDict is None:
        shapeDict = {key: 2 for key in rawKeys}
    return {key + suffix: create_trivial_core(key + suffix, shapeDict[key], [key]) for key in rawKeys}


def create_trivial_core(name, shape, colors):
    return engine.get_core()(np.ones(shape), colors, name)


def create_random_core(name, shape, colors, randomEngine="NumpyUniform"):
    if randomEngine == "NumpyUniform":
        return engine.get_core()(np.random.random(size=shape), colors, name)
    elif randomEngine == "NumpyNormal":
        return engine.get_core()(np.random.normal(size=shape), colors, name)
    else:
        raise ValueError("Random Engine {} not known for core creation!".format(randomEngine))

def create_basis_core(name, shape, colors, numberTuple):
    values = np.zeros(shape=shape)
    values[numberTuple] = 1
    return engine.get_core()(values, colors, name)