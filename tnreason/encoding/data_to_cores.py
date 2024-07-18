from tnreason import engine
import numpy as np

dataCoreSuffix = "_dataCore"
defaultDataColor = "j"


def create_data_cores(sampleDf, atomKeys=None, dataColor=defaultDataColor, interpretation="atomic", dimensionsDict={},
                      coreType="NumpyTensorCore"):
    """
    Creates a tensor network of data cores, each storing the by atomKey selected column of sampleDf as a core of
    the CP Decomposition of the one-hot encoding (empirical distribution)
    """
    if atomKeys is None:
        atomKeys = list(sampleDf.columns[1:])
    if coreType == "NumpyTensorCore":
        if interpretation == "atomic":
            return {
                atomKey + dataCoreSuffix: atomValues_from_sampleDf(sampleDf, atomKey, dataColor)
                for atomKey in atomKeys if atomKey in list(sampleDf.columns)}
        elif interpretation == "categorical":
            dimensionsDict.update({catKey: 2 for catKey in atomKeys if catKey not in dimensionsDict})
            return {
                catKey + dataCoreSuffix: catValues_from_sampleDf(sampleDf, catKey, dataColor, dimensionsDict[catKey])
                for
                catKey in atomKeys
            }
    elif coreType == "PolynomialCore":  # Then directly taking categorical interpretation
        slices = []
        for i, row in sampleDf.iterrows():
            slices.append([i, {catKey: int(row[catKey]) for catKey in atomKeys}])
        return {"polynomial" + dataCoreSuffix: engine.get_core(coreType)(values=engine.SliceValues(slices, shape=[]),
                                                                         colors=atomKeys,
                                                                         name="polynomial" + dataCoreSuffix)}
    else:
        raise ValueError("Interpretation {} not understood!".format(interpretation))


def atomValues_from_sampleDf(sampleDf, atomKey, dataColor):
    dataNum = sampleDf.values.shape[0]
    dfEntries = sampleDf[atomKey].values
    values = np.zeros(shape=(dataNum, 2))
    for i in range(dataNum):
        values[i, 1] = dfEntries[i]
        values[i, 0] = 1 - dfEntries[i]
    return engine.get_core()(values, [dataColor, atomKey], name=atomKey + dataCoreSuffix)


def catValues_from_sampleDf(sampleDf, catKey, dataColor, catDim):
    dataNum = sampleDf.values.shape[0]
    dfEntries = sampleDf[catKey].values
    values = np.zeros(shape=(dataNum, catDim))
    for i in range(dataNum):
        values[i, int(dfEntries[i])] = 1
    return engine.get_core()(values, [dataColor, catKey], name=catKey + dataCoreSuffix)
