from tnreason.encoding.formulas import create_formulas_cores, create_raw_formula_cores, get_formula_color
from tnreason.encoding.auxiliary import create_emptyCoresDict, get_variables, get_all_variables
from tnreason.encoding.storage import save_as_yaml, load_from_yaml

def get_head_core(expression, headType, weight=None, coreType="NumpyTensorCore", name=None):
    import tnreason.encoding.formulas as enform
    return enform.create_headCore(expression=expression, headType=headType, weight=weight, coreType=coreType, name=name)


def get_constraint_cores(constraintsDict):
    import tnreason.encoding.constraints as encon
    return encon.create_constraints(constraintsDict)

def get_trivial_cores(variableList, coreType="NumpyTensorCore", varDimDict=None):
    import tnreason.encoding.auxiliary as enaux
    return enaux.create_emptyCoresDict(variableList, coreType=coreType, varDimDict=varDimDict)

def get_neuron_cores(name, connectiveList, candidatesDict):
    import tnreason.encoding.neurons as enneur
    return enneur.create_neuron(name, connectiveList, candidatesDict)


def get_architecture_cores(specDict):
    import tnreason.encoding.neurons as enneur
    return enneur.create_architecture(specDict)


def load_architecture_cores(loadPath):
    import tnreason.encoding.storage as stor
    return get_architecture_cores(stor.load_from_yaml(loadPath))


def get_knowledge_cores(specDict):
    import tnreason.encoding.formulas as enform
    import tnreason.encoding.constraints as encon

    if "weightedFormulas" in specDict.keys():
        knowledgeCores = enform.create_formulas_cores(specDict["weightedFormulas"])
        if "facts" in specDict.keys():
            knowledgeCores = {**knowledgeCores,
                              **enform.create_formulas_cores(specDict["facts"], list(knowledgeCores.keys()))}
    elif "facts" in specDict.keys():
        knowledgeCores = enform.create_formulas_cores(specDict["facts"])
    else:
        knowledgeCores = {}

    if "categoricalConstraints" in specDict.keys():
        knowledgeCores = {**knowledgeCores, **encon.create_constraints(specDict["categoricalConstraints"])}

    return knowledgeCores

