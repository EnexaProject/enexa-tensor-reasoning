from tnreason import engine
from tnreason import encoding

from tnreason.knowledge import distributions as dist

import numpy as np


class EntropyMaximizer:
    """
    Optimizes the weights of weighted formulas based on an entropy maximization principle.
    Independent implementation of the special case of two-dimensional exponentiated weights of algorithms.MomentMatcher
    """

    def __init__(self, expressionsDict, satisfactionDict, backCores={},
                 contractionMethod=engine.defaultContractionMethod):
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
        factsDict = {}
        if updateKeys is None:
            updateKeys = list(self.satisfactionDict.keys())
        for optKey in updateKeys:
            if self.satisfactionDict[optKey] == 0:
                updateKeys.remove(optKey)
                self.formulaCores.update(
                    encoding.create_head_core(self.expressionsDict[optKey], headType="falseEvaluation"))
                print("Formula {} is never satisfied.".format(self.expressionsDict[optKey]))
                factsDict[optKey] = 0
            elif self.satisfactionDict[optKey] == 1:
                updateKeys.remove(optKey)
                self.formulaCores.update(
                    encoding.create_head_core(self.expressionsDict[optKey], headType="truthEvaluation"))
                print("Formula {} is always satisfied.".format(self.expressionsDict[optKey]))
                factsDict[optKey] = 1
        weightDict = {key: [] for key in updateKeys}
        for sweep in range(sweepNum):
            for optKey in updateKeys:
                local_weight = self.local_condition_satisfier(optKey, self.satisfactionDict[optKey])
                weightDict[optKey].append(local_weight)
                self.formulaCores.update(encoding.create_head_core(self.expressionsDict[optKey], headType="expFactor",
                                                                   weight=local_weight))
        return weightDict, factsDict

    def local_condition_satisfier(self, optKey, empRate):
        optColor = encoding.get_formula_color(self.expressionsDict[optKey])
        tboCoreKey = optColor + encoding.headCoreSuffix
        negValue, posValue = engine.contract(method=self.contractionMethod,
                                             coreDict={**self.backCores,
                                                       **{key: self.formulaCores[key] for key in self.formulaCores if
                                                          key != tboCoreKey},
                                                       tboCoreKey: encoding.create_trivial_core(tboCoreKey,
                                                                        self.formulaCores[tboCoreKey].values.shape,
                                                                        self.formulaCores[tboCoreKey].colors)
                                                       },
                                             openColors=[optColor]).values
        if negValue != 0 and posValue != 0:
            return np.log((negValue / posValue) * (empRate / (1 - empRate)))
        elif negValue == 0 or posValue == 0:
            return 0  ## In this case the formula is redundant

    def get_optimized_knowledge_base(self, sweepNum=10, updateKeys=None):
        weightDict, factsDict = self.alternating_optimization(sweepNum=sweepNum, updateKeys=updateKeys)
        return dist.HybridKnowledgeBase(
            weightedFormulas={key: self.expressionsDict[key] + [weightDict[key][-1]] for key in weightDict},
            facts = {key : self.expressionsDict[key] for key in factsDict},
            backCores = self.backCores
        )
