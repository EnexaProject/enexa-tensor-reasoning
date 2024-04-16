from tnreason import engine
from tnreason import encoding

import numpy as np


class EntropyMaximizer:
    def __init__(self, expressionsDict, satisfactionDict, backCores, contractionMethod="NumpyEinsum"):
        self.backCores = backCores
        self.expressionsDict = expressionsDict
        self.satisfactionDict = satisfactionDict

        self.formulaCores = encoding.create_formulas_cores({
            key: [expressionsDict[key], 0] for key in expressionsDict
        })

        self.contractionMethod = contractionMethod

    def alternating_optimization(self, sweepNum=10, updateKeys=None):
        if updateKeys is None:
            updateKeys = list(self.satisfactionDict.keys())
        for optKey in updateKeys:
            if self.satisfactionDict[optKey] == 0:
                updateKeys.remove(optKey)
                self.formulaCores.update(
                    encoding.create_headCore(self.expressionsDict[optKey], headType="falseEvaluation"))
                print("Formula {} is never satisfied.".format(self.expressionsDict[optKey]))
            elif self.satisfactionDict[optKey] == 1:
                updateKeys.remove(optKey)
                self.formulaCores.update(
                    encoding.create_headCore(self.expressionsDict[optKey], headType="truthEvaluation"))
                print("Formula {} is always satisfied.".format(self.expressionsDict[optKey]))

        solutionDict = {key: [] for key in updateKeys}
        for sweep in range(sweepNum):
            for optKey in updateKeys:
                local_weight = self.local_condition_satisfier(optKey, self.satisfactionDict[optKey])
                solutionDict[optKey].append(local_weight)
                self.formulaCores.update(encoding.create_headCore(self.expressionsDict[optKey], headType="expFactor",
                                                                  weight=local_weight))
        return solutionDict

    def local_condition_satisfier(self, optKey, empRate):
        optColor = encoding.get_formula_color(self.expressionsDict[optKey])
        negValue, posValue = engine.contract(method=self.contractionMethod,
                                             coreDict={**self.backCores,
                                                       **{key: self.formulaCores[key] for key in self.formulaCores if
                                                          key != optColor + "_headCore"}}, openColors=[optColor]).values

        if negValue != 0 and posValue != 0:
            return np.log((negValue / posValue) * (empRate / (1 - empRate)))
        elif negValue == 0 or posValue == 0:
            return 0  ## In this case the formula is redundant


class EmpiricalDistribution:
    def __init__(self, sampleDf, atomKeys=None):
        self.create_from_sampleDf(sampleDf, atomKeys)

    def create_from_sampleDf(self, sampleDf, atomKeys=None):
        if atomKeys is None:
            atomKeys = list(sampleDf.columns)
        self.dataCores = encoding.create_data_cores(sampleDf, atomKeys)
        self.dataNum = sampleDf.values.shape[0]

    def get_empirical_satisfaction(self, expression):
        return engine.contract(method="NumpyEinsum",
                               coreDict={**self.dataCores, **encoding.create_raw_formula_cores(expression)},
                               openColors=[encoding.get_formula_color(expression)]).values[1] / self.dataNum

    def get_satisfactionDict(self, expressionsDict):
        return {key: self.get_empirical_satisfaction(expressionsDict[key]) for key in expressionsDict}
