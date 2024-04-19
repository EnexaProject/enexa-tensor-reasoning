from tnreason import engine
from tnreason.encoding import connectives

import numpy as np

connectiveKey = "connectives"
candidatesKey = "candidates"

connectiveSelColorSuffix = "_actVar"
connectiveSelCoreSuffix = "_actCore"

candidatesColorSuffix = "_selVar"
candidatesCoreSuffix = "_selCore"

posPrefix = "p"


def create_architecture(specDict):
    architectureCores = {}
    for neuronName in specDict.keys():
        architectureCores = {**architectureCores,
                             **create_neuron(neuronName, specDict[neuronName][connectiveKey], {
                                 posPrefix + str(i): posCandidates for i, posCandidates in
                                 enumerate(specDict[neuronName][candidatesKey])
                             })}
    return architectureCores


def create_solution_expression(specDict, solutionDict):
    fixedNeurons = fix_neurons(specDict, solutionDict)
    headNeurons = get_headKeys(fixedNeurons)

    if len(headNeurons) != 1:
        print("WARNING: Headneurons are {}.".format(headNeurons))
    return {headKey: replace_neuronnames(headKey, fixedNeurons) for headKey in headNeurons}


def replace_neuronnames(currentNeuronName, fixedNeurons):
    if currentNeuronName not in fixedNeurons:
        return currentNeuronName  ## Then an atom
    currentNeuron = fixedNeurons[currentNeuronName]
    arity = len(currentNeuron[1])
    if arity == 1:
        return [currentNeuron[0], replace_neuronnames(currentNeuron[1][0], fixedNeurons)]
    elif arity == 2:
        return [replace_neuronnames(currentNeuron[1][0], fixedNeurons), currentNeuron[0],
                replace_neuronnames(currentNeuron[1][1], fixedNeurons)]
    else:
        raise ValueError("Neuronname {} has unsuported arity {}!".format(currentNeuronName, arity))


def get_headKeys(fixedNeurons):
    headKeys = set(fixedNeurons.keys())
    for formulaKey in fixedNeurons:
        for inNeuron in fixedNeurons[formulaKey][1]:
            if inNeuron in headKeys:
                headKeys.remove(inNeuron)
    return headKeys


def fix_neurons(specDict, solutionDict):
    rawFormulas = {}
    for neuronName in specDict:
        rawFormulas[neuronName] = [
            specDict[neuronName][connectiveKey][solutionDict[neuronName + connectiveSelColorSuffix]],
            [specDict[neuronName][candidatesKey][i][
                 solutionDict[neuronName + "_p" + str(i) + candidatesColorSuffix]] for i in
             range(len(specDict[neuronName][candidatesKey]))]
            ]
    return rawFormulas


def find_atoms(specDict):
    atoms = set()
    for neuronName in specDict.keys():
        for positionList in specDict[neuronName][candidatesKey]:
            atoms = atoms | set(positionList)
    return list(atoms)


def find_selection_dimDict(specDict):
    dimDict = {}
    for neuronName in specDict:
        dimDict.update({neuronName + connectiveSelColorSuffix: len(specDict[neuronName][connectiveKey]),
                        **{neuronName + "_p" + str(i) + candidatesColorSuffix: len(candidates)
                           for i, candidates in enumerate(specDict[neuronName][candidatesKey])}})
    return dimDict


def create_neuron(name, connectiveList, candidatesDict={}):
    neuronCores = {
        name + connectiveSelCoreSuffix: create_connective_selectors(name, candidatesDict.keys(), connectiveList)}
    for candidateKey in candidatesDict:
        neuronCores = {**neuronCores, **create_variable_selectors(
            name, candidateKey, candidatesDict[candidateKey])}
    return neuronCores


def create_variable_selectors(
        neuronName, candidateKey, variables, coreType="NumpyTensorCore"):
    cSelectorDict = {}
    for i, variableKey in enumerate(variables):
        values = np.ones(
            shape=(len(variables), 2, 2))  # control, atom (subexpression), formulatruth (headexpression)
        values[i, 0, 1] = 0
        values[i, 1, 0] = 0

        cSelectorDict[neuronName + "_" + candidateKey + "_" + variableKey + candidatesCoreSuffix] = engine.get_core(
            coreType)(
            values,
            [neuronName + "_" + candidateKey + candidatesColorSuffix,
             variableKey,
             candidateKey],
            neuronName + "_" + candidateKey + "_" + variableKey + candidatesCoreSuffix
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

    return engine.get_core(coreType)(values, [neuronName + connectiveSelColorSuffix, *candidateKeys, neuronName])
