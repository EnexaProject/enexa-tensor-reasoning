from tnreason import engine
from tnreason.encoding import connectives

import numpy as np


def create_architecture(specDict):
    architectureCores = {}
    for neuronName in specDict.keys():
        architectureCores = {**architectureCores,
                             **create_neuron(neuronName, specDict[neuronName]["connectiveList"], {
                                 "p" + str(i): posCandidates for i, posCandidates in
                                 enumerate(specDict[neuronName]["candidatesList"])
                             })}
    return architectureCores


def create_neuron(name, connectiveList, candidatesDict={}):
    neuronCores = {name + "_conCore": create_connective_selectors(name, candidatesDict.keys(), connectiveList)}
    for candidateKey in candidatesDict:
        neuronCores = {**neuronCores, **create_variable_selectors(
            name, candidateKey, candidatesDict[candidateKey])}
    return neuronCores


def create_variable_selectors(
        neuronName, candidateKey, variables, coreType="NumpyTensorCore"):
    cSelectorDict = {}
    for i, variableKey in enumerate(variables):
        values = np.ones(shape=(len(variables), 2, 2))  # control, atom (subexpression), formulatruth (headexpression)
        values[i, 0, 1] = 0
        values[i, 1, 0] = 0

        cSelectorDict[neuronName + "_" + candidateKey + "_" + variableKey + "_selCore"] = engine.get_core(coreType)(
            values,
            [neuronName + "_" + candidateKey + "_selVar",
             variableKey,
             candidateKey],
            neuronName + "_" + candidateKey + "_" + variableKey + "_selCore"
        )
    return cSelectorDict


def create_connective_selectors(neuronName, candidateKeys, connectiveList, coreType="NumpyTensorCore"):
    if len(candidateKeys) == 1:
        values = np.empty((len(connectiveList), 2, 2))
        for i, connectiveKey in enumerate(connectiveList):
            values[i] = connectives.get_unary_tensor(connectiveKey)

    elif len(candidateKeys) == 2:
        values = np.empty((len(connectiveList), 2, 2, 2))
        for i, connectiveKey in enumerate(connectiveList):
            values[i] = connectives.get_binary_tensor(connectiveKey)
    else:
        raise ValueError("Number of candidates wrong in Neuron {}!".format(neuronName))

    return engine.get_core(coreType)(values, [neuronName + "_conVar", *candidateKeys, neuronName])
