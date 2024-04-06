def create_evidenceCoresDict(evidenceDict, dimDict=None,coreType="NumpyTensorCore"):
    if dimDict is None:
        dimDict = {evidenceKey : 2 for evidenceKey in evidenceDict}
    evidenceCoresDict = {}
    for atomKey in evidenceDict:
        truthValues = np.zeros(shape=(dimDict[atomKey]))
        if bool(evidenceDict[atomKey]):
            truthValues[1] = 1
        else:
            truthValues[0] = 1
        evidenceCoresDict[atomKey + "_evidence"] = tensor.get_core(coreType=coreType)(truthValues, [atomKey],
                                                                                      atomKey + "_evidence")
    return evidenceCoresDict
