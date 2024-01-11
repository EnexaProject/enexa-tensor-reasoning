import numpy as np

from tnreason.contraction import core_contractor as coc
from tnreason.model import tensor_model as tm
from tnreason.model import formula_tensors as ft


def expected_cross_entropy(testExpressionsDict, generativeExpressionsDict):
    expTestTensorModel = tm.TensorRepresentation(testExpressionsDict, headType="expFactor")
    nonexpTestTensorModel = tm.TensorRepresentation(testExpressionsDict, headType="truthEvaluation")
    expGenerativeTensorModel = tm.TensorRepresentation(generativeExpressionsDict, headType="expFactor")

    testPartition = expTestTensorModel.contract_partition()
    generativePartition = expGenerativeTensorModel.contract_partition()

    crossTerm = coc.CoreContractor({**nonexpTestTensorModel.all_cores(),
                                    **expGenerativeTensorModel.all_cores()}).contract().values

    return np.log(testPartition) - crossTerm / generativePartition


def expected_shannon_entropy(testExpressionsDict):
    return expected_cross_entropy(testExpressionsDict, testExpressionsDict)


def expected_KL_divergence(testExpressionsDict, generativeExpressionsDict):
    return expected_cross_entropy(testExpressionsDict, generativeExpressionsDict) - expected_shannon_entropy(
        generativeExpressionsDict)


## Further entropies:
# empirical_cross_entropy: computed by the likelihood in MLE Base
# empirical_KL_divergence: difference of likelihood with empirical shannon entropy (also done in MLE Base)

def empirical_shannon_entropy(sampleDf, atoms=None):
    ## The Shannon entropy of the empirical distribution
    dataNum = sampleDf.values.shape[0]
    if atoms is None:
        atoms = sampleDf.columns

    dataCores = {atomKey: ft.dataCore_from_sampleDf(sampleDf, atomKey) for atomKey in atoms}
    contracted = coc.CoreContractor(dataCores, openColors=atoms).contract().multiply(1 / dataNum)
    ## Again suffering from the curse of dimensionality!

    logContracted = contracted.clone()
    logContracted.values = np.log(contracted.values)
    ## Remove -infty, since causing problems
    # but just appearing on zero data coordinates (thus not contributing in contraction)
    logContracted.values[logContracted.values < -1e308] = 0

    return -coc.CoreContractor({"data": contracted, "log": logContracted}).contract().values