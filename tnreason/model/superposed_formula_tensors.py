from tnreason.logic import expression_utils as eu
from tnreason.logic import coordinate_calculus as cc

from tnreason.contraction import generic_cores as gc

from tnreason.model import create_cores as crc

import numpy as np


class SuperPosedFormulaTensor:
    def __init__(self, skeletonExpression, candidatesDict, parameterCoresDict={}, name="spfTensor",
                 coreType="CoordinateCore"):
        self.skeletonExpression, replaceDict = eu.replace_double_symbols(skeletonExpression, {})
        self.coresDict = parameterCoresDict

        self.skeletonExpression = skeletonExpression
        self.candidatesDict = candidatesDict
        self.name = name

        self.symbols = eu.get_symbols(
            self.skeletonExpression)  ## Symbols are the placeholderKeys + _0 for each repetition
        for symbol in self.symbols:
            if symbol.split("_")[
                0] not in candidatesDict:  ## Adding placeHolders to candidatesDict (here they need to be atoms or connectives)
                candidatesDict[symbol] = [symbol.split("_")[0]]
        self.create_cores(coreType)

    def create_cores(self, coreType):
        for symbol in self.symbols:
            self.coresDict = {**self.coresDict,
                              **self.create_local_cores(symbol, coreType)}

    def create_local_cores(self, symbol, coreType="CoordinateCore"):
        placeHolderKey = symbol.split("_")[0]
        phType = eu.decide_symbol_type(self.skeletonExpression, symbol)
        if phType == "binary":
            subExpression = eu.get_subexpression(self.skeletonExpression, symbol)
            upColor = self.name + "_" + str(subExpression)
            leftColor = self.name + "_" + str(subExpression[0])
            rightColor = self.name + "_" + str(subExpression[2])
            if coreType == "NumpyTensorCore":
                return {symbol + "_binSelector":
                            gc.NumpyTensorCore(values=create_binary_values(self.candidatesDict[placeHolderKey]),
                                               colors=[placeHolderKey, leftColor, rightColor, upColor],
                                               name=symbol + "_binSelector")}
            elif coreType == "CoordinateCore":
                return {symbol + "_binSelector":
                            cc.CoordinateCore(create_binary_values(self.candidatesDict[placeHolderKey]),
                                              [placeHolderKey, leftColor, rightColor, upColor],
                                              name=symbol + "_binSelector")}
        elif phType == "unary":
            subExpression = eu.get_subexpression(self.skeletonExpression, symbol)
            upColor = self.name + "_" + str(subExpression)
            downColor = self.name + "_" + str(subExpression[1])
            if coreType == "NumpyTensorCore":
                return {symbol + "_uniSelector":
                            gc.NumpyTensorCore(values=create_unary_values(self.candidatesDict[placeHolderKey]),
                                               colors=[placeHolderKey, downColor, upColor],
                                               name=symbol + "_uniSelector")}
            elif coreType == "CoordinateCore":
                return {symbol + "_uniSelector":
                            cc.CoordinateCore(create_unary_values(self.candidatesDict[placeHolderKey]),
                                              [placeHolderKey, downColor, upColor],
                                              name=symbol + "_uniSelector")}
        elif phType == "atom":
            return create_controlled_atom_selectors(self.candidatesDict[placeHolderKey], symbol, self.name,
                                                    coreType)

        else:
            raise ValueError("Placeholder {} not in the expression {}!".format(symbol, self.skeletonExpression))

    def get_cores(self):
        return self.coresDict


def create_controlled_atom_selectors(atoms, symbol, formulaKey, coreType="NumpyTensorCore"):
    placeHolder = symbol.split("_")[0]
    if len(atoms) == 1 and atoms[
        0] == placeHolder:  ## The Case of single atom in candidate - Nothing to controlled select
        return {}
    cSelectorDict = {}
    for i, atomKey in enumerate(atoms):
        values = np.ones(shape=(len(atoms), 2, 2))  # control, atom (subexpression), formulatruth (headexpression)
        values[i, 0, 1] = 0
        values[i, 1, 0] = 0
        if coreType == "NumpyTensorCore":
            cSelectorDict[symbol + "_" + atomKey + "_selector"] = gc.NumpyTensorCore(values=values,
                                                                                     colors=[placeHolder, atomKey,
                                                                                             formulaKey + "_" + symbol],
                                                                                     name=symbol + "_" + atomKey + "_selector")
        elif coreType == "CoordinateCore":
            cSelectorDict[symbol + "_" + atomKey + "_selector"] = cc.CoordinateCore(values,
                                                                                    [placeHolder, atomKey,
                                                                                     formulaKey + "_" + symbol],
                                                                                    symbol + "_" + atomKey + "_selector")
    return cSelectorDict


def create_unary_values(symbols):
    coreValues = np.empty(shape=(len(symbols), 2, 2))
    for i, symbol in enumerate(symbols):
        coreValues[i] = crc.get_unary_tensor(symbol)
    return coreValues


def create_binary_values(symbols):
    coreValues = np.empty(shape=(len(symbols), 2, 2, 2))
    for i, symbol in enumerate(symbols):
        coreValues[i] = crc.get_binary_tensor(symbol)
    return coreValues


if __name__ == "__main__":
    spfTensor = SuperPosedFormulaTensor(["jaszczur", "piskle", ["sledz", "sikorka"]],
                                        {
                                            "piskle": ["and", "or", "imp"],
                                            # "not1" : ["not"],
                                            # "sledz": ["alphasledz"],
                                            "sikorka": ["sikorka12"],
                                            "sledz": ["not"],
                                            "jaszczur": ["a1", "a2", "a3"]
                                        })
    # print(spfTensor.get_cores().keys())
    cores = spfTensor.get_cores()
    for key in cores:
        print(key, cores[key].values.shape, cores[key].colors)
        # print(cores[key].values)

    from tnreason.contraction import core_contractor as coc

    print(
        coc.CoreContractor(coreDict=spfTensor.get_cores(),
                           openColors=["sikorka"]).contract().values)
