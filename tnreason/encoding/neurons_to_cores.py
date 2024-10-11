from tnreason import engine

from tnreason.encoding import connectives as con
from tnreason.encoding import formulas_to_cores as enform

import numpy as np

connectiveSelColorSuffix = "_actVar"
connectiveSelCoreSuffix = "_actCore"

candidatesColorSuffix = "_selVar"
candidatesCoreSuffix = "_selCore"

posPrefix = "p"


def create_architecture(neuronDict, headNeurons=[]):
    """
    Creates a tensor network of neuron cores with selection colors
        * neuronDict: Dictionary specifying to each neuronName a list of candidates (for the connective and the arguments)
        * headNeurons: List of neuronNames to be associated with hard headCores
    """
    architectureCores = {}
    for neuronName in neuronDict.keys():
        architectureCores = {**architectureCores,
                             **create_neuron(neuronName, neuronDict[neuronName][0], {
                                 neuronName + "_" + posPrefix + str(i): posCandidates for i, posCandidates in
                                 enumerate(neuronDict[neuronName][1:])
                             })}
    for headNeuron in headNeurons:
        architectureCores = {**architectureCores, **enform.create_head_core(headNeuron, headType="truthEvaluation")}
    return architectureCores


def create_neuron(neuronName, connectiveList, candidatesDict={}):
    """
    Creates the cores to one neuron 
        * neuronName: String to use as prefix of the key to each core
        * connectiveList: List of connectives to be selected
        * candidatesDict: Dictionary of lists of candidates to each argument of the neuron
    """
    neuronCores = {
        neuronName + connectiveSelCoreSuffix: create_connective_selectors(neuronName, candidatesDict.keys(),
                                                                          connectiveList)}
    for candidateKey in candidatesDict:
        neuronCores = {**neuronCores, **create_variable_selectors(
            neuronName, candidateKey, candidatesDict[candidateKey])}
    return neuronCores


def create_variable_selectors(neuronName, candidateKey, variables):
    """
    Creates the selection cores to one argument at a neuron.
    There are two possibilities to specify variables
        * list of variables string: Representing a selection of atomic variables represented in the string and a CP decomposition is created.
        * single string: Representing a categorical variable in the format X=[m] and a single selection core is created.
    Resulting colors in each core: [selection variable, candidate variable, neuron argument variable]
    """
    if isinstance(variables, str):
        catName, dimBracket = variables.split("=")
        dim = int(dimBracket.split("[")[1][:-1])
        values = np.zeros(shape=(dim, dim, 2))
        values[:, :, 0] = np.ones(shape=(dim, dim)) - np.eye(dim)
        values[:, :, 1] = np.eye(dim)
        return {candidateKey + "_" + variables + candidatesCoreSuffix:
                    engine.get_core()(values,
                                      [candidateKey + candidatesColorSuffix, catName, candidateKey],
                                      candidateKey + "_" + variables + candidatesCoreSuffix)
                }

    cSelectorDict = {}
    for i, variableKey in enumerate(variables):
        values = np.ones(
            shape=(len(variables), 2, 2))  # control, atom (subexpression), formulatruth (headexpression)
        values[i, 0, 1] = 0
        values[i, 1, 0] = 0

        cSelectorDict[candidateKey + "_" + variableKey + candidatesCoreSuffix] = engine.get_core()(
            values,
            [candidateKey + candidatesColorSuffix,
             variableKey,
             candidateKey],
            candidateKey + "_" + variableKey + candidatesCoreSuffix
        )
    return cSelectorDict


def create_connective_selectors(neuronName, candidateKeys, connectiveList):
    """
    Creates the connective selection core, using the candidateKeys as color and arity specification
    """
    if len(candidateKeys) == 1:
        return engine.create_relational_encoding(inshape=[len(connectiveList), 2], outshape=[2],
                                                 incolors=[neuronName + connectiveSelColorSuffix, *candidateKeys],
                                                 outcolors=[neuronName],
                                                 function=con.get_unary_connective_selector(connectiveList),
                                                 name=neuronName+connectiveSelCoreSuffix)
    elif len(candidateKeys) == 2:
        return engine.create_relational_encoding(inshape=[len(connectiveList), 2, 2], outshape=[2],
                                                 incolors=[neuronName + connectiveSelColorSuffix, *candidateKeys],
                                                 outcolors=[neuronName],
                                                 function=con.get_binary_connective_selector(connectiveList),
                                                 name=neuronName+connectiveSelCoreSuffix)
    else:
        raise ValueError(
            "Number {} of candidates wrong in Neuron {} with connectives {}!".format(len(candidateKeys), neuronName,
                                                                                     connectiveList))

## Functions to identify solution expressions when candidates are selected
def create_solution_expression(neuronDict, selectionDict):
    """
    Replaces the candidates of neurons by solutions and returns the identified head neurons as formulas
        * neuronDict: Dictionary specifying the neurons
        * selectionDict: Dictionary selecting candidates (connective and position) to each selection variables at each neuron
    """
    fixedNeurons = fix_neurons(neuronDict, selectionDict)
    headNeurons = get_headKeys(fixedNeurons)
    if len(headNeurons) != 1:
        print("WARNING: Headneurons are {}.".format(headNeurons))
    return {headKey: replace_neuronnames(headKey, fixedNeurons) for headKey in headNeurons}


def fix_neurons(neuronDict, selectionDict):
    """
    Replaces the neurons with subexpressions refering to each other
    """
    rawFormulas = {}
    for neuronName in neuronDict:
        rawFormulas[neuronName] = [neuronDict[neuronName][0][selectionDict[neuronName + connectiveSelColorSuffix]]] + [
            fix_selection(neuronDict[neuronName][i],
                          selectionDict[neuronName + "_" + posPrefix + str(i - 1) + candidatesColorSuffix])
            for i in range(1, len(neuronDict[neuronName]))]
    return rawFormulas


def fix_selection(choices, position):
    """
    Materializes a choice, either from a categorical variable (when choices is str) or from a list of possibilities (when choices is a list of str)
    """
    if isinstance(choices, str):  # The case of a categorical variable
        return choices + "=" + str(position)
    else:  # The case of a list of possibilities
        return choices[position]


def get_headKeys(fixedNeurons):
    """
    Identifies the independent neurons as heads
    """
    headKeys = set(fixedNeurons.keys())
    for formulaKey in fixedNeurons:
        for inNeuron in fixedNeurons[formulaKey][1:]:
            if inNeuron in headKeys:
                headKeys.remove(inNeuron)
    return headKeys


def replace_neuronnames(currentNeuronName, fixedNeurons):
    """
    Replaces the current neuron with the respective expression, after iterative replacement of depending fixed neurons
    """
    if currentNeuronName not in fixedNeurons:
        return currentNeuronName  ## Then an atom
    currentNeuron = fixedNeurons[currentNeuronName].copy()
    currentNeuron = [currentNeuron[0]] + [replace_neuronnames(currentNeuron[i], fixedNeurons) for i in
                                          range(1, len(currentNeuron))]
    return currentNeuron


## Auxiliary functions for knowledge identifying the atoms and the dimension of selection variables
def find_atoms(specDict):
    atoms = set()
    for neuronName in specDict.keys():
        for positionList in specDict[neuronName][1:]:
            atoms = atoms | set(positionList)
    return list(atoms)


def find_selection_dimDict(specDict):
    dimDict = {}
    for neuronName in specDict:
        dimDict.update({neuronName + connectiveSelColorSuffix: len(specDict[neuronName][0]),
                        **{neuronName + "_" + posPrefix + str(i) + candidatesColorSuffix: len(candidates)
                           for i, candidates in enumerate(specDict[neuronName][1:])}})
    return dimDict

def find_selection_colors(specDict):
    """
    Extracts the default selection colors from a architecture dict
    """
    colors = []
    for neuronName in specDict:
        colors.append(neuronName + connectiveSelColorSuffix)
        colors = colors + [neuronName + "_" + posPrefix + str(i) + candidatesColorSuffix for i in range(len(specDict[neuronName][1:]))]
    return colors