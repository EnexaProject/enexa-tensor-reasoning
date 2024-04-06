from tnreason import engine

import numpy as np


def create_emptyCoresDict(variableList, coreType="NumpyTensorCore", varDimDict=None):
    if varDimDict is None:
        varDimDict = {variableKey: 2 for variableKey in variableList}
    return {variableKey + "_trivialCore": engine.get_core(coreType=coreType)(np.ones(varDimDict[variableKey]),
                                                                             [variableKey],
                                                                             variableKey + "_trivialCore")
            for variableKey in variableList}
