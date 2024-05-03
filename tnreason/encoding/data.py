from tnreason import engine
import numpy as np

dataCoreSuffix = "_dataCore"


def create_data_cores(sampleDf, atomKeys=None, coreType="NumpyTensorCore", dataColor="j"):
    if atomKeys is None:
        atomKeys = list(sampleDf.columns[1:])
    return {
        atomKey + dataCoreSuffix: atomValues_from_sampleDf(sampleDf, atomKey, dataColor, coreType=coreType)
        for atomKey in atomKeys if atomKey in list(sampleDf.columns)}


def atomValues_from_sampleDf(sampleDf, atomKey, dataColor, coreType="NumpyTensorCore"):
    dataNum = sampleDf.values.shape[0]
    dfEntries = sampleDf[atomKey].values
    values = np.zeros(shape=(dataNum, 2))
    for i in range(dataNum):
        values[i, 1] = dfEntries[i]
        values[i, 0] = 1 - dfEntries[i]
    return engine.get_core(coreType=coreType)(values, [dataColor, atomKey], name=atomKey + dataCoreSuffix)
