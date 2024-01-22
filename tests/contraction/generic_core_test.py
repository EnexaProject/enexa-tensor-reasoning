from tnreason.contraction import core_contractor as coc
from tnreason.contraction import generic_cores as gc

from tnreason.logic import coordinate_calculus as cc

import numpy as np

coreDict = {
    "c1": cc.CoordinateCore(np.random.binomial(10, 0.5, size=(3, 2)), ["a", "b"]),
    "c2": cc.CoordinateCore(np.random.binomial(20, 0.8, size=(3, 2, 5)), ["a", "b", "c"])
}

tensorCoreDict = {
    key: gc.change_type(coreDict[key]) for key in coreDict
}

ccContractor = coc.CoreContractor(coreDict,
                                  openColors=["a", "b"])

tcContractor = coc.ChainTensorContractor(tensorCoreDict,
                                        openColors=["a", "b"]
                                        )

print(tcContractor.contract().colors)
print(ccContractor.contract().colors)
print(np.linalg.norm(tcContractor.contract().values - ccContractor.contract().values))
