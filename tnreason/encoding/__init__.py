def get_formula_cores(expression, alreadyCreated=[]):
    import tnreason.encoding.formulas as enform
    return enform.create_conCores(expression, alreadyCreated=alreadyCreated)


def get_formulas_cores(expressionsDict, alreadyCreated=[]):
    import tnreason.encoding.formulas as enform
    return enform.create_formulas(expressionsDict, alreadyCreated=alreadyCreated)


def get_head_core(expression, headType, weight=None, coreType="NumpyTensorCore", name=None):
    import tnreason.encoding.formulas as enform
    return enform.create_headCore(expression=expression, headType=headType, weight=weight, coreType=coreType, name=name)


def get_constraint_cores(constraintsDict, alreadyCreated=[]):
    import tnreason.encoding.constraints as encon
    return encon.create_constraints(constraintsDict)


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
        knowledgeCores = enform.create_formulas(specDict["weightedFormulas"])
        if "facts" in specDict.keys():
            knowledgeCores = {**knowledgeCores,
                              **enform.create_formulas(specDict["facts"], list(knowledgeCores.keys()))}
    elif "facts" in specDict.keys():
        knowledgeCores = enform.create_formulas(specDict["facts"])
    else:
        knowledgeCores = {}

    if "categoricalConstraints" in specDict.keys():
        knowledgeCores = {**knowledgeCores, **encon.create_constraints(specDict["categoricalConstraints"])}

    return knowledgeCores


if __name__ == "__main__":
    print(get_formula_cores([["a", "imp", "b"], "or", "c"], alreadyCreated=['(a_imp_b)_conCore']))

    print(get_neuron_cores(
        "funneur", connectiveList=["not"],
        candidatesDict={"sledz": ["jaszczur", "sikorka"]}
    ))
