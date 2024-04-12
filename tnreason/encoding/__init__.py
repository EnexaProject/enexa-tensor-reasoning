def get_formula_cores(expression, alreadyCreated=[]):
    import tnreason.encoding.formulas as enform
    return enform.create_conCores(expression, alreadyCreated=alreadyCreated)


def get_head_core(color, headType, weight=None, coreType="NumpyTensorCore", name=None):
    import tnreason.encoding.formulas as enform
    return enform.create_headCore(color=color, headType=headType, weight=weight, coreType=coreType, name=name)


def get_neuron_cores(name, connectiveList, candidatesDict):
    import tnreason.encoding.neurons as enneur
    return enneur.create_neuron(name, connectiveList, candidatesDict)

def get_architecture_cores(specDict):
    import tnreason.encoding.neurons as enneur
    return enneur.create_architecture(specDict)

def load_architecture_cores(loadPath):
    import tnreason.encoding.storage as stor
    return get_architecture_cores(stor.load_from_yaml(loadPath))




if __name__ == "__main__":
    print(get_formula_cores([["a", "imp", "b"], "or", "c"], alreadyCreated=['(a_imp_b)_conCore']))

    print(get_neuron_cores(
        "funneur", connectiveList=["not"],
        candidatesDict={"sledz": ["jaszczur", "sikorka"]}
    ))
