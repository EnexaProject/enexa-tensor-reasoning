from tnreason.logic import expression_utils as eu
from tnreason.logic import coordinate_calculus as cc

from tnreason.model import create_cores as crc

import numpy as np


class SuperPosedFormulaTensor:
    def __init__(self, skeletonExpression, candidatesDict, parameterCoresDict={}, name="spfTensor"):
        self.skeletonExpression, replaceDict = eu.replace_double_symbols(skeletonExpression, {})
        self.coresDict = {}

        self.skeletonExpression = skeletonExpression
        self.candidatesDict = candidatesDict
        self.name = name

        self.symbols = eu.get_symbols(
            self.skeletonExpression)  ## Symbols are the placeholderKeys + _0 for each repetition
        for symbol in self.symbols:
            if symbol.split("_")[
                0] not in candidatesDict:  ## Adding placeHolders to candidatesDict (here they need to be atoms or connectives)
                candidatesDict[symbol] = [symbol.split("_")[0]]
        self.create_cores()

    def create_cores(self):
        for symbol in self.symbols:
            self.coresDict = {**self.coresDict,
                              **self.create_local_cores(symbol)}

    def create_local_cores(self, symbol):
        placeHolderKey = symbol.split("_")[0]
        phType = eu.decide_symbol_type(self.skeletonExpression, symbol)
        if phType == "binary":
            subExpression = eu.get_subexpression(self.skeletonExpression, symbol)
            upColor = self.name + "_" + str(subExpression)
            leftColor = self.name + "_" + str(subExpression[0])
            rightColor = self.name + "_" + str(subExpression[2])

            return {symbol + "_binSelector":
                        cc.CoordinateCore(create_binary_core(self.candidatesDict[placeHolderKey]),
                                          core_colors=[placeHolderKey, upColor, leftColor, rightColor])}
        elif phType == "unary":
            subExpression = eu.get_subexpression(self.skeletonExpression, symbol)
            upColor = self.name + "_" + str(subExpression)
            downColor = self.name + "_" + str(subExpression[1])
            return {symbol + "_uniSelector":
                        cc.CoordinateCore(create_unary_core(self.candidatesDict[placeHolderKey]),
                                          [placeHolderKey, upColor, downColor])}
        elif phType == "atom":
            return crc.create_local_selectorCores(self.candidatesDict[placeHolderKey], symbol)

        else:
            raise ValueError("Placeholder {} not in the expression {}!".format(symbol, self.skeletonExpression))

    def get_cores(self):
        return self.coresDict


def create_unary_core(symbols):
    coreValues = np.empty(shape=(len(symbols), 2, 2))
    for i, symbol in enumerate(symbols):
        coreValues[i] = crc.get_unary_tensor(symbol)
    return coreValues


def create_binary_core(symbols):
    coreValues = np.empty(shape=(len(symbols), 2, 2, 2))
    for i, symbol in enumerate(symbols):
        coreValues[i] = crc.get_binary_tensor(symbol)
    return coreValues


if __name__ == "__main__":
    spfTensor = SuperPosedFormulaTensor(["jaszczur", "jaszczur", ["sledz", "sikorka"]],
                                        {
                                            # "not1" : ["not"],
                                            # "sledz": ["alphasledz"],
                                            "sledz": ["not"],
                                            "jaszczur": ["and", "or", "imp"]
                                        })
    # print(spfTensor.get_cores().keys())
    cores = spfTensor.get_cores()
    for key in cores:
        print(key, cores[key].values.shape, cores[key].colors)
        print(cores[key].values)
