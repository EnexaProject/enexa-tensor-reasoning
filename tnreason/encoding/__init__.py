from tnreason.encoding.formulas import create_formulas_cores, create_raw_formula_cores, get_formula_color, create_headCore
from tnreason.encoding.constraints import create_constraints
from tnreason.encoding.auxiliary import create_emptyCoresDict, get_variables, get_all_variables
from tnreason.encoding.storage import save_as_yaml, load_from_yaml
from tnreason.encoding.neurons import create_neuron, create_architecture
from tnreason.encoding.data import create_data_cores
def load_architecture_cores(loadPath):
    import tnreason.encoding.storage as stor
    return create_architecture(stor.load_from_yaml(loadPath))

def create_evidence_cores(evidenceDict):
    return create_formulas_cores({**{key: key for key in evidenceDict if evidenceDict[key]},
            **{key: ["not", key] for key in evidenceDict if not evidenceDict[key]}
            })

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

