from tnreason import engine

dataCoreSuffix = "_dataCore"
defaultDataColor = "j"



def create_data_cores(sampleDf, atomKeys=None, dataColor=defaultDataColor, interpretation="atomic", dimensionsDict=None,
                      coreType=None, partitionDict=None):
    """
    Creates a tensor network of data cores, each storing the by atomKey selected column of sampleDf as a core of
    the CP Decomposition of the one-hot encoding (empirical distribution)
    """
    if atomKeys is None:
        atomKeys = list(sampleDf.columns)
    if dimensionsDict is None:
        dimensionsDict = {atomKey: 2 for atomKey in atomKeys}
    if interpretation == "categorical":
        return categorical_to_relational_encoding(sampleDf, atomKeys, dataColor, dimensionsDict=dimensionsDict,
                                                  coreType=coreType, partitionDict=partitionDict)
    elif interpretation == "atomic":
        return {atomKey + dataCoreSuffix: atomValues_from_sampleDf(sampleDf, atomKey, dataColor)
                for atomKey in atomKeys if atomKey in list(sampleDf.columns)}
    else:
        raise ValueError("Interpretation {} not understood!".format(interpretation))

def categorical_to_relational_encoding(sampleDf, atomKeys=None, dataColor=defaultDataColor, dimensionsDict=None,
                                       coreType=None, partitionDict=None):
    """
    Relational Encoding of samples, which are interpreted as certain states of categorical variables.
    """
    if atomKeys is None:
        atomKeys = list(sampleDf.columns)
    if dimensionsDict is None:
        dimensionsDict = {atomKey: 2 for atomKey in atomKeys}
    return engine.create_partitioned_relational_encoding(
        inshape=[sampleDf.values.shape[0]], outshape=[dimensionsDict[atomKey] for atomKey in atomKeys],
        incolors=[dataColor], outcolors=atomKeys,
        function= lambda k: sampleDf[atomKeys].iloc[k].values,
        coreType=coreType,
        nameSuffix=dataCoreSuffix,
        partitionDict=partitionDict
    )


def atomValues_from_sampleDf(sampleDf, atomKey, dataColor, coreType=None):
    """
    Tensor Encoding of samples, which are interpreted as atomic uncertainties.
    """
    dataNum = sampleDf.values.shape[0]
    dfEntries = sampleDf[atomKey].values
    tensorFunc = lambda j, a: (1 - a) * (1 - dfEntries[int(j)]) + a * dfEntries[int(j)]
    return engine.create_tensor_encoding(inshape=[dataNum, 2], incolors=[dataColor, atomKey], function=tensorFunc,
                                         coreType=coreType,
                                         name=atomKey + dataCoreSuffix)