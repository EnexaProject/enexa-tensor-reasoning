from tnreason.model import  formula_tensors as ft

from tnreason.model import generate_test_data as gtd

from tnreason.contraction import core_contractor as coc

import numpy as np

def alternative_compute_shannon_entropy(dTensor):
    contractedData = coc.CoreContractor(dTensor.get_cores(),
                                        openColors=dTensor.atoms).contract().values.flatten() / dTensor.dataNum
    logContractedData = np.log(np.copy(contractedData))
    logContractedData[logContractedData < -1e308] = 0
    return -np.dot(logContractedData, contractedData)

sampleDf = gtd.generate_sampleDf({
        "f1": [["sikorka", "and", ["not","piskle"]], 2],
        "f2": [[["not","sledz"], "and", ["not","szczeniak"]], 20],
        "f3": [["jaszczur", "and", "sikorka"], 2],
    }, 10)

dTensor = ft.DataTensor(sampleDf)

print(dTensor.compute_shannon_entropy())
print(alternative_compute_shannon_entropy(dTensor))

#print(dTensor.efficient_shan non_contraction())