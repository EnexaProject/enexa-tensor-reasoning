from tnreason import engine
import numpy as np

dataCoreSuffix = "_dataCore"
defaultDataColor = "j"

def samples_to_map(samples, variableList):
    return lambda k: samples[variableList].iloc[k].values

def categorical_to_relational_encoding(sampleDf, atomKeys=None, dataColor=defaultDataColor, dimensionsDict=None,
                                       coreType=engine.defaultCoreType, partitionDict=None):
    if atomKeys is None:
        atomKeys = list(sampleDf.columns)
    if dimensionsDict is None:
        dimensionsDict = {atomKey : 2 for atomKey in atomKeys}
    return engine.create_partitioned_relational_encoding(
        inshape=[sampleDf.values.shape[0]], outshape=[dimensionsDict[atomKey] for atomKey in atomKeys],
        incolors=[dataColor], outcolors=atomKeys,
        function=samples_to_map(sampleDf, atomKeys),
        coreType=coreType,
        nameSuffix=dataCoreSuffix,
        partitionDict=partitionDict
    )

def create_data_cores(sampleDf, atomKeys=None, dataColor=defaultDataColor, interpretation="atomic", dimensionsDict=None,
                      coreType=engine.defaultCoreType, partitionDict=None):
    """
    Creates a tensor network of data cores, each storing the by atomKey selected column of sampleDf as a core of
    the CP Decomposition of the one-hot encoding (empirical distribution)
    """
    if atomKeys is None:
        atomKeys = list(sampleDf.columns)
    if dimensionsDict is None:
        dimensionsDict = {atomKey : 2 for atomKey in atomKeys}
    if interpretation=="categorical":
        return categorical_to_relational_encoding(sampleDf, atomKeys, dataColor, dimensionsDict=dimensionsDict,
                                                  coreType=coreType, partitionDict=partitionDict)
    elif interpretation == "atomic": # Then only coreType == "NumpyTensorCore" supported
            return {atomKey + dataCoreSuffix: atomValues_from_sampleDf(sampleDf, atomKey, dataColor)
                for atomKey in atomKeys if atomKey in list(sampleDf.columns)}
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