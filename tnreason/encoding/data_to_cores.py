from tnreason import engine
import numpy as np

dataCoreSuffix = "_dataCore"
defaultDataColor = "j"


def create_data_cores(sampleDf, atomKeys=None, dataColor=defaultDataColor):
    """
    Creates a tensor network of data cores, each storing the by atomKey selected column of sampleDf as a core of
    the CP Decomposition of the one-hot encoding (empirical distribution)
    """
    if atomKeys is None:
        atomKeys = list(sampleDf.columns[1:])
    return {
        atomKey + dataCoreSuffix: atomValues_from_sampleDf(sampleDf, atomKey, dataColor)
        for atomKey in atomKeys if atomKey in list(sampleDf.columns)}


def atomValues_from_sampleDf(sampleDf, atomKey, dataColor):
    dataNum = sampleDf.values.shape[0]
    dfEntries = sampleDf[atomKey].values
    values = np.zeros(shape=(dataNum, 2))
    for i in range(dataNum):
        values[i, 1] = dfEntries[i]
        values[i, 0] = 1 - dfEntries[i]
    return engine.get_core()(values, [dataColor, atomKey], name=atomKey + dataCoreSuffix)
