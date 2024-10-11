from tnreason.engine.engine_visualization import draw_factor_graph
from tnreason.engine.polynomial_contractor import SliceValues

from tnreason.engine.auxiliary_cores import create_trivial_cores, create_trivial_core, create_basis_core

from tnreason.engine.contraction_handling import contract, defaultContractionMethod

from tnreason.engine.creation_handling import get_core, defaultCoreType, create_tensor_encoding, \
    create_relational_encoding, create_partitioned_relational_encoding, create_random_core


def get_dimDict(coreDict):
    dimDict = {}
    for coreKey in coreDict:
        dimDict.update({color: coreDict[coreKey].values.shape[i] for i, color in enumerate(coreDict[coreKey].colors)})
    return dimDict
