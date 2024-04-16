from tnreason import engine

import numpy as np


def create_data_cores(sampleDf, atomKeys, atomColorDict=None, coreType="NumpyTensorCore", dataColor="j"):
    if atomColorDict is None:
        atomColorDict = {atomKey: atomKey for atomKey in atomKeys}
    return {atomKey + "_dataCore": dataCore_from_sampleDf(sampleDf, atomColorDict[atomKey], dataColor, coreType=coreType)
            for atomKey in atomKeys}


## DataCore Creation
def dataCore_from_sampleDf(sampleDf, atomKey, dataColor, coreType="NumpyTensorCore"):
    if atomKey not in sampleDf.keys():
        raise ValueError
    dfEntries = sampleDf[atomKey].values
    dataNum = dfEntries.shape[0]
    values = np.zeros(shape=(dataNum, 2))
    for i in range(dataNum):
        if dfEntries[i] == 0:
            values[i, 0] = 1
        else:
            values[i, 1] = 1
    return engine.get_core(coreType=coreType)(values, [dataColor, atomKey])
