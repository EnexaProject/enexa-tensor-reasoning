import numpy as np

from tnreason.contraction import core_contractor as coc
from tnreason.model import tensor_model as tm
from tnreason.model import formula_tensors as ft


def expected_cross_entropy(testExpressionsDict, generativeExpressionsDict):
    testTensorModel = tm.TensorRepresentation(testExpressionsDict, headType="expFactor")
    expGenerativeTensorModel = tm.TensorRepresentation(generativeExpressionsDict, headType="expFactor")

    testPartition = testTensorModel.contract_partition()
    generativePartition = expGenerativeTensorModel.contract_partition()

    crossTerm = np.sum(
        coc.CoreContractor({**testTensorModel.get_cores([formulaKey], headType="weightedTruthEvaluation"),
            **expGenerativeTensorModel.all_cores()}).contract().values
        for formulaKey in testExpressionsDict
    )

    return np.log(testPartition) - crossTerm / generativePartition


def expected_shannon_entropy(testExpressionsDict):
    return expected_cross_entropy(testExpressionsDict, testExpressionsDict)


def expected_KL_divergence(testExpressionsDict, generativeExpressionsDict):
    return expected_cross_entropy(testExpressionsDict, generativeExpressionsDict) - expected_shannon_entropy(
        generativeExpressionsDict)


## Further entropies:
# empirical_cross_entropy: computed by the likelihood in MLE Base
# empirical_shannon_entropy: in ft.DataTensor
# empirical_KL_divergence: difference of likelihood with empirical shannon entropy (also done in MLE Base)

def empirical_shannon_entropy(sampleDf, atoms=None):
    return ft.DataTensor(sampleDf,atoms).compute_shannon_entropy()