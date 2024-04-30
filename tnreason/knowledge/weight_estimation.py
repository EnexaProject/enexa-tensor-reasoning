from tnreason import engine
from tnreason import encoding

import numpy as np


class EntropyMaximizer:
    def __init__(self, expressionsDict, satisfactionDict, backCores={}, contractionMethod="NumpyEinsum"):
        self.backCores = backCores
        self.expressionsDict = expressionsDict
        self.satisfactionDict = satisfactionDict
        for key in self.satisfactionDict:
            assert self.satisfactionDict[key] <= 1 and self.satisfactionDict[
                key] >= 0, "Empirical satisfaction rate {} of key {} is wrong.".format(self.satisfactionDict[key], key)
        self.formulaCores = encoding.create_formulas_cores({
            key: expressionsDict[key] + [0] for key in expressionsDict
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