import numpy as np

from tnreason.contraction import core_contractor as coc
from tnreason.model import tensor_model as tm


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



#def empirical_cross_entropy(testExpressionsDict, sampleDf):
## This is the likelihood, computed in MLEBase not here.

def empirical_shannon_entropy(sampleDf):
    ## Contract datacores with itself, i.e. norm of the datacore?
    pass

#def empirical_KL_divergence(testExpressionsDict, sampleDf):
#    return empirical_cross_entropy(testExpressionsDict, sampleDf) - empirical_shannon_entropy(sampleDf)
