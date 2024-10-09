from tnreason.engine.engine_visualization import draw_factor_graph
from tnreason.engine.polynomial_contractor import SliceValues

from tnreason.engine.auxiliary_cores import create_trivial_cores, create_trivial_core, create_random_core, \
    create_basis_core

from tnreason.engine.contraction_handling import contract, defaultContractionMethod

defaultCoreType = "NumpyTensorCore"
def get_core(coreType="NumpyTensorCore"):
    if coreType == "NumpyTensorCore":
        from tnreason.engine.workload_to_numpy import NumpyCore
        return NumpyCore
    elif coreType == "PolynomialCore":
        from tnreason.engine.polynomial_contractor import PolynomialCore
        return PolynomialCore
    else:
        raise ValueError("Core Type {} not supported.".format(coreType))

def get_dimDict(coreDict):
    dimDict = {}
    for coreKey in coreDict:
        dimDict.update({color: coreDict[coreKey].values.shape[i] for i, color in enumerate(coreDict[coreKey].colors)})
    return dimDict