from tnreason import contraction

from tnreason.logic import coordinate_calculus as cc

import numpy as np


defaultContractionMethod = "PgmpyVariableEliminator"

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
                   **create_evidenceCores(self.assignment.keys(), self.assignment, self.sampleDimDict),
                   sampleKey + "_trivial": create_trivialCore(sampleKey, self.sampleDimDict[sampleKey])}

        unNormalized = contraction.get_contractor(contractionMethod)(conDict,
                                                                     openColors=[sampleKey]).contract().values
        prob = unNormalized / np.sum(unNormalized)
        self.assignment[sampleKey] = np.where(np.random.multinomial(1, prob) == 1)[0][0]


def create_trivialCore(varKey, varDim):
    return cc.CoordinateCore(np.ones(varDim), [varKey], varKey + "_trivial")


def create_evidenceCores(varKeys, valueDict, dimDict):
    return {varKey + "_evidence": create_evidenceCore(varKey, valueDict[varKey], dimDict[varKey]) for varKey in varKeys}


def create_evidenceCore(varKey, varValue, varDim):
    values = np.zeros(varDim)
    values[varValue] = 1
    return cc.CoordinateCore(values, [varKey], varKey + "_evidence")
