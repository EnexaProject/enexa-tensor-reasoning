from tnreason.logic import expression_utils as eu
from tnreason.logic import coordinate_calculus as cc

from tnreason.model import create_cores as crc

import numpy as np


class SuperPosedFormulaTensor:
    def __init__(self, skeletonExpression, candidatesDict, parameterCoresDict={}):
        self.skeletonExpression = skeletonExpression
        self.candidatesDict = candidatesDict

        symbols = eu.get_symbols(self.skeletonExpression)
        for symbol in symbols:
            if symbol not in candidatesDict:
                candidatesDict[symbol] = [symbol]
        self.create_cores()

    def create_cores(self):
        self.coresDict = {}
        for placeHolderKey in self.candidatesDict:
            self.coresDict = {**self.coresDict,
                              **self.create_local_cores(placeHolderKey)}

    def create_local_cores(self, placeHolderKey):
        phType = eu.decide_symbol_type(self.skeletonExpression, placeHolderKey)
        if phType == "binary":
            subExpression = eu.get_binary_subexpression(self.skeletonExpression, placeHolderKey)
            upColor = placeHolderKey + "_" + str(subExpression)
            leftColor = placeHolderKey + "_" + str(subExpression[1])
            rightColor = placeHolderKey + "_" + str(subExpression[2])

            return {placeHolderKey + "_binSelector":
                        cc.CoordinateCore(create_binary_core(self.candidatesDict[placeHolderKey]),
                                          core_colors=[placeHolderKey, upColor, leftColor, rightColor])}
        elif phType == "unary":
            return {}
        elif phType == "atom":
            return crc.create_local_selectorCores(self.candidatesDict[placeHolderKey], placeHolderKey)

    def get_cores(self):
        return self.coresDict


def create_binary_core(symbols):
    coreValues = np.empty(shape=(len(symbols), 2, 2, 2))
    for i, symbol in enumerate(symbols):
        coreValues[i] = crc.get_binary_tensor(symbol)
    return coreValues


if __name__ == "__main__":
    spfTensor = SuperPosedFormulaTensor(["sledz", "jaszczur", "sikorka"],
                                        {
                                            "sledz": ["alphasledz"],
                                            "jaszczur": ["and", "or"]
                                        })
    print(spfTensor.get_cores().keys())
