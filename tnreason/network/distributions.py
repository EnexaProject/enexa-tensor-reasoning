from tnreason import contraction

from tnreason.tensor import model_cores as mcore

import numpy as np

defaultContractionMethod = "PgmpyVariableEliminator"
defaultCoreType = "NumpyTensorCore"


class TNDistribution:
    def __init__(self, distributionCores):
        self.distributionCores = distributionCores

    def heat_cores(self, coreKeys):
        ## coreKeys should be HeadCores
        pass

    def gibbs_sampling(self, sampleKeys, sampleDimDict, sweepNum=10, contractionMethod=defaultContractionMethod):
        self.assignment = {}  ## sampleKey, value
        self.sampleDimDict = sampleDimDict  ## stores leg dimensions of sampleKeys

        for sweep in range(sweepNum):
            for sampleKey in sampleKeys:
                self.gibbs_step(sampleKey, contractionMethod)

        return self.assignment

    def gibbs_step(self, sampleKey, contractionMethod):
        if sampleKey in self.assignment:
            self.assignment.pop(sampleKey)
        conDict = {**self.distributionCores,
                   **mcore.create_evidenceCoresDict(self.assignment, dimDict=self.sampleDimDict,
                                                    coreType=defaultCoreType),
                   **mcore.create_emptyCoresDict([sampleKey], varDimDict={sampleKey: self.sampleDimDict[sampleKey]},
                                                 coreType=defaultCoreType)}

        unNormalized = contraction.get_contractor(contractionMethod)(conDict,
                                                                     openColors=[sampleKey]).contract().values
        prob = unNormalized / np.sum(unNormalized)
        self.assignment[sampleKey] = np.where(np.random.multinomial(1, prob) == 1)[0][0]
