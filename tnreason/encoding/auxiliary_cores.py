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
    return {key + suffix: engine.get_core()(np.ones(shapeDict[key]), [key], key + suffix)
            for key in rawKeys}
